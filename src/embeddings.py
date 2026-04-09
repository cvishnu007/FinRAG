"""
src/embeddings.py  — updated for FinBERT support
=================================================
Supports multiple encoder backends:
    - all-MiniLM-L6-v2        (fast, general, dim=384)
    - yiyanghkust/finbert-tone (financial, dim=768)
    - ProsusAI/finbert         (financial, dim=768)

Install:
    pip install sentence-transformers transformers torch
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict, cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def _encode_with_sentence_transformers(
    texts: list,
    model_name: str,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    """Use sentence-transformers library — works for MiniLM and finbert-tone."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def _encode_with_transformers(
    texts: list,
    model_name: str,
    batch_size: int,
    show_progress: bool,
) -> np.ndarray:
    """
    Use HuggingFace transformers directly — needed for ProsusAI/finbert
    which has a classification head that sentence-transformers handles
    less cleanly. Extracts [CLS] token embedding from last hidden state.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    all_embeddings = []
    iterator = range(0, len(texts), batch_size)

    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="Batches")

    with torch.no_grad():
        for start in iterator:
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            output = model(**encoded)

            # Use [CLS] token — first token of last hidden state
            cls_embeddings = output.last_hidden_state[:, 0, :]

            # L2 normalise for cosine similarity
            norms = cls_embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cls_embeddings = cls_embeddings / norms

            all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


# ── Model registry ────────────────────────────────────────────────────────────
# Maps model name → (encoder_function, expected_dim)
MODEL_REGISTRY = {
    "all-MiniLM-L6-v2"          : (_encode_with_sentence_transformers, 384),
    "yiyanghkust/finbert-tone"   : (_encode_with_sentence_transformers, 768),
    "ProsusAI/finbert"           : (_encode_with_transformers, 768),
    "paraphrase-mpnet-base-v2"   : (_encode_with_sentence_transformers, 768),
    "all-mpnet-base-v2"          : (_encode_with_sentence_transformers, 768),
}


def encode_texts(
    texts: pd.Series,
    cache_path: str = "artifacts/cache/embeddings.pkl",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a pandas Series of strings into dense embeddings.
    Cache is keyed by (model_name, text_hash) so switching models
    automatically uses a separate cache — no collisions.

    Returns
    -------
    np.ndarray of shape (len(texts), dim)
    where dim=384 for MiniLM and dim=768 for FinBERT variants
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}\n"
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    encoder_fn, expected_dim = MODEL_REGISTRY[model_name]

    texts = texts.fillna("").astype(str).reset_index(drop=True)

    # Key cache entries by BOTH model name and text hash to avoid collisions
    # when switching between models
    def make_key(text: str) -> str:
        return f"{model_name}::{_hash_text(text)}"

    keys = texts.apply(make_key)

    cache = _load_cache(cache_path)

    missing_mask  = ~keys.isin(cache)
    missing_texts = texts[missing_mask].tolist()

    if missing_texts:
        if show_progress:
            print(f"  [{model_name}] Encoding {len(missing_texts):,} new texts "
                  f"({(~missing_mask).sum():,} from cache)...")

        new_embeddings = encoder_fn(
            missing_texts, model_name, batch_size, show_progress
        )

        for k, emb in zip(keys[missing_mask], new_embeddings):
            cache[k] = emb

        _save_cache(cache, cache_path)
        if show_progress:
            print(f"  Cache saved → {cache_path}")
    else:
        if show_progress:
            print(f"  [{model_name}] All {len(texts):,} texts from cache.")

    embeddings = np.stack([cache[k] for k in keys])
    return embeddings