"""
src/embeddings.py
=================
Sentence embedding encoder with on-disk caching.
Uses all-MiniLM-L6-v2 via sentence-transformers.

Install:
    pip install sentence-transformers
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd


def _hash_text(text: str) -> str:
    """MD5 hash of a string — used as cache key."""
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


def encode_texts(
    texts: pd.Series,
    cache_path: str = "artifacts/cache/embeddings.pkl",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a pandas Series of strings into 384-dim embeddings.

    - Texts already in the cache are returned instantly.
    - Only unseen texts are sent to the model.
    - Cache is saved after encoding.

    Returns
    -------
    np.ndarray of shape (len(texts), 384)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        )

    texts = texts.fillna("").astype(str).reset_index(drop=True)
    hashes = texts.apply(_hash_text)

    cache = _load_cache(cache_path)

    # Find which indices need encoding
    missing_mask = ~hashes.isin(cache)
    missing_texts = texts[missing_mask].tolist()

    if missing_texts:
        if show_progress:
            print(f"  Encoding {len(missing_texts):,} new texts "
                  f"({(~missing_mask).sum():,} served from cache)...")

        model = SentenceTransformer(model_name)
        new_embeddings = model.encode(
            missing_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine sim = dot product
        )

        # Store in cache
        for h, emb in zip(hashes[missing_mask], new_embeddings):
            cache[h] = emb

        _save_cache(cache, cache_path)
        if show_progress:
            print(f"  Cache saved → {cache_path}")
    else:
        if show_progress:
            print(f"  All {len(texts):,} texts served from cache.")

    # Assemble output in original order
    embeddings = np.stack([cache[h] for h in hashes])
    return embeddings  # shape: (n, 384)