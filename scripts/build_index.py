"""
scripts/build_index.py
======================
One-time script to build the FAISS retrieval index over all
training-set events using cached MiniLM embeddings.

Run:
    python -m scripts.build_index
"""

import os
import numpy as np
import pandas as pd

from src.embeddings import encode_texts
from src.retrieval import build_index, load_index, query, format_retrieved_events
from src.features import build_feature_frame, build_text_column
from src.market_features import add_market_features

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/processed/master_dataset.csv"
PRICES_DIR   = "data/raw/prices"
TIMEZONE     = "America/New_York"
MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"
CACHE_PATH   = "artifacts/cache/embeddings.pkl"
LABEL_THRESHOLD = 0.002
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15


def time_split_per_ticker(df, train_frac=0.70, val_frac=0.15):
    train_parts, val_parts, test_parts = [], [], []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("published").reset_index(drop=True)
        n = len(group)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train : n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val :])
    return (
        pd.concat(train_parts).sort_values("published").reset_index(drop=True),
        pd.concat(val_parts).sort_values("published").reset_index(drop=True),
        pd.concat(test_parts).sort_values("published").reset_index(drop=True),
    )


def main():
    # ── Load and prepare data — identical pipeline to train_blend.py ──────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "label"])
    df = df.sort_values("published").reset_index(drop=True)

    # Same label filter as train_blend.py
    before = len(df)
    df = df[df["return_1d"].abs() > LABEL_THRESHOLD].copy()
    print(f"Label filter: {before:,} → {len(df):,} rows")

    # Same dedup as train_blend.py
    before = len(df)
    df = df.sort_values(["published", "ticker"])
    df = df.drop_duplicates(subset=["title", "published"], keep="first")
    df = df.sort_values("published").reset_index(drop=True)
    print(f"Cross-ticker dedup: {before:,} → {len(df):,} rows")

    # Same split as train_blend.py — index is built on TRAIN ONLY
    # We must never index val or test events — that would be leakage
    train_df, val_df, test_df = time_split_per_ticker(
        df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )
    print(f"Train split: {len(train_df):,} events → these go into the index")
    print(f"Val:  {len(val_df):,}  |  Test: {len(test_df):,}  (NOT indexed)")

    # ── Build text column ─────────────────────────────────────────────────────
    train_df = train_df.copy()
    train_df["text"] = build_text_column(train_df)

    # ── Encode training texts ─────────────────────────────────────────────────
    # Most will be served from cache — only new ones get encoded
    print("\nEncoding training texts (most from cache)...")
    train_embeddings = encode_texts(
        train_df["text"],
        cache_path=CACHE_PATH,
        show_progress=True,
    )
    print(f"  Embeddings shape: {train_embeddings.shape}")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    print("\nBuilding FAISS index...")
    build_index(
        embeddings=train_embeddings,
        metadata_df=train_df,
    )

    # ── Smoke test — query with a known training example ──────────────────────
    print("\n── Smoke test ───────────────────────────────────────────────")
    print("Loading index back from disk and running a test query...")

    index, metadata = load_index()
    print(f"Index loaded: {index.ntotal:,} vectors")

    # Pick a random training example as query
    sample = train_df.sample(1, random_state=42).iloc[0]
    sample_text = pd.Series([sample["text"]])
    sample_emb  = encode_texts(sample_text, cache_path=CACHE_PATH, show_progress=False)

    print(f"\nQuery article:")
    print(f"  Ticker : {sample['ticker']}")
    print(f"  Title  : {sample['title'][:100]}")
    print(f"  Return : {sample['return_1d']*100:+.2f}%")

    # Retrieve top 5 — excluding same ticker to show cross-ticker power
    results = query(
        embedding=sample_emb[0],
        index=index,
        metadata=metadata,
        k=5,
        exclude_ticker=sample["ticker"],
    )

    print("\nTop 5 similar historical events (cross-ticker):")
    print(format_retrieved_events(results))

    # Also show same-ticker retrieval
    results_same = query(
        embedding=sample_emb[0],
        index=index,
        metadata=metadata,
        k=5,
        exclude_ticker=None,
    )
    print("\nTop 5 similar historical events (including same ticker):")
    print(format_retrieved_events(results_same))

    print("\n✅ Index build complete. Ready for explanation generation.")
    print("   Next: python -m scripts.run_explain")


if __name__ == "__main__":
    main()