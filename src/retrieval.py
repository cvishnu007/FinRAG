"""
src/retrieval.py
================
FAISS-based retrieval index for historical financial news events.
Embeddings must be L2-normalised (encode_texts does this by default).

Install:
    pip install faiss-cpu
"""

from __future__ import annotations

import os
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd


# ── Default paths ─────────────────────────────────────────────────────────────
INDEX_PATH    = "artifacts/retrieval/faiss_index.pkl"
METADATA_PATH = "artifacts/retrieval/metadata.pkl"


def build_index(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    index_path: str = INDEX_PATH,
    metadata_path: str = METADATA_PATH,
) -> None:
    """
    Build a FAISS flat inner-product index from training embeddings.

    Parameters
    ----------
    embeddings   : (n, 384) float32 array — L2-normalised MiniLM embeddings
    metadata_df  : DataFrame with columns ticker, title, summary,
                   published, trade_date, return_1d, label
    """
    try:
        import faiss
    except ImportError:
        raise RuntimeError(
            "faiss-cpu is required. Install: pip install faiss-cpu"
        )

    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    emb = embeddings.astype(np.float32)
    dim = emb.shape[1]

    # IndexFlatIP = exact inner product search
    # Since embeddings are L2-normalised, inner product == cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    print(f"  FAISS index built: {index.ntotal:,} vectors, dim={dim}")

    # Serialise to bytes for pickling
    index_bytes = faiss.serialize_index(index).tobytes()
    with open(index_path, "wb") as f:
        pickle.dump(index_bytes, f)
    print(f"  Index saved → {index_path}")

    # Store metadata aligned to index positions
    keep_cols = [c for c in [
        "ticker", "title", "summary", "published",
        "trade_date", "return_1d", "excess_return", "label"
    ] if c in metadata_df.columns]

    meta = metadata_df[keep_cols].reset_index(drop=True)
    with open(metadata_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  Metadata saved → {metadata_path} ({len(meta):,} rows)")


def load_index(
    index_path: str = INDEX_PATH,
    metadata_path: str = METADATA_PATH,
):
    """
    Load the FAISS index and metadata from disk.

    Returns
    -------
    index    : faiss.Index
    metadata : pd.DataFrame
    """
    try:
        import faiss
    except ImportError:
        raise RuntimeError(
            "faiss-cpu is required. Install: pip install faiss-cpu"
        )

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            f"Run: python -m scripts.build_index"
        )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path}. "
            f"Run: python -m scripts.build_index"
        )

    with open(index_path, "rb") as f:
        index_bytes = pickle.load(f)

    index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def query(
    embedding: np.ndarray,
    index,
    metadata: pd.DataFrame,
    k: int = 5,
    exclude_ticker: str = None,
) -> List[Dict]:
    """
    Retrieve the top-K most similar historical events for a query embedding.

    Parameters
    ----------
    embedding      : (384,) or (1, 384) float32 array — L2-normalised
    index          : loaded FAISS index
    metadata       : aligned metadata DataFrame
    k              : number of results to return
    exclude_ticker : optionally retrieve cross-ticker analogues only
                     (set to the query ticker to exclude same-stock results)

    Returns
    -------
    List of dicts, each containing:
        ticker, title, summary, published, trade_date,
        return_1d, label, similarity
    """
    emb = np.array(embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)

    # Search more than k if we need to filter by ticker
    search_k = k * 5 if exclude_ticker else k

    similarities, indices = index.search(emb, search_k)
    similarities = similarities[0]   # shape (search_k,)
    indices      = indices[0]        # shape (search_k,)

    results = []
    for sim, idx in zip(similarities, indices):
        if idx < 0 or idx >= len(metadata):
            continue

        row = metadata.iloc[idx]

        if exclude_ticker and row["ticker"] == exclude_ticker:
            continue

        results.append({
            "ticker"      : row["ticker"],
            "title"       : row["title"],
            "summary"     : row.get("summary", ""),
            "published"   : str(row["published"])[:10],   # date only
            "trade_date"  : str(row["trade_date"]),
            "return_1d"   : float(row["return_1d"]),
            "label"       : int(row["label"]),
            "similarity"  : float(sim),
        })

        if len(results) >= k:
            break

    return results


def format_retrieved_events(events: List[Dict]) -> str:
    """
    Format retrieved events into a readable string for LLM context.
    """
    if not events:
        return "No similar historical events found."

    lines = []
    for i, e in enumerate(events, 1):
        direction = "UP ✅" if e["label"] == 1 else "DOWN ❌"
        ret_pct   = e["return_1d"] * 100
        lines.append(
            f"{i}. {e['ticker']} — \"{e['title'][:120]}\"\n"
            f"   Date: {e['published']}  |  "
            f"Next-day return: {ret_pct:+.2f}%  |  {direction}\n"
            f"   Similarity: {e['similarity']:.3f}"
        )

    # Summary stats
    avg_ret  = sum(e["return_1d"] for e in events) / len(events) * 100
    up_count = sum(1 for e in events if e["label"] == 1)

    lines.append(
        f"\nSummary: {up_count}/{len(events)} events were UP | "
        f"Average return: {avg_ret:+.2f}%"
    )

    return "\n".join(lines)