"""
scripts/run_explain.py
======================
End-to-end demo: loads the blend model + FAISS index, picks
10 random test examples, generates predictions + explanations.

Run:
    set GROQ_API_KEY=your_key_here     (Windows)
    python -m scripts.run_explain
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from src.embeddings import encode_texts
from src.retrieval import load_index
from src.explain import explain_prediction
from src.features import build_feature_frame, build_text_column
from src.market_features import add_market_features
from src.utils import ensure_dir

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "data/processed/master_dataset.csv"
PRICES_DIR      = "data/raw/prices"
TIMEZONE        = "America/New_York"
MARKET_OPEN     = "09:30"
MARKET_CLOSE    = "16:00"
CACHE_PATH      = "artifacts/cache/embeddings.pkl"
LABEL_THRESHOLD = 0.002
TRAIN_FRAC      = 0.70
VAL_FRAC        = 0.15
N_EXAMPLES      = 10
SEED            = 42

MARKET_COLS    = ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]
SENTIMENT_COLS = ["pos_ratio", "neg_ratio", "sent_score", "surprise_score"]
CAT_COLS       = ["ticker", "dow", "month", "session", "earnings_season"]


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

def _extract_first_col(X):
    # NOTE: Replace this body with the exact code from your training script!
    # Usually it looks something like one of these:
    if hasattr(X, "iloc"):
        return X.iloc[:, 0:1]
    return X[:, 0:1]

def main():
    # ── Load data — same pipeline as train_blend.py ───────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "label"])
    df = df.sort_values("published").reset_index(drop=True)

    before = len(df)
    df = df[df["return_1d"].abs() > LABEL_THRESHOLD].copy()
    print(f"Label filter: {before:,} → {len(df):,} rows")

    before = len(df)
    df = df.sort_values(["published", "ticker"])
    df = df.drop_duplicates(subset=["title", "published"], keep="first")
    df = df.sort_values("published").reset_index(drop=True)
    print(f"Cross-ticker dedup: {before:,} → {len(df):,} rows")

    df = add_market_features(
        df, prices_dir=PRICES_DIR, timezone=TIMEZONE, market_close=MARKET_CLOSE
    )

    df = build_feature_frame(
        df, timezone=TIMEZONE,
        market_open=MARKET_OPEN, market_close=MARKET_CLOSE,
        use_sentiment=True,
    )

    _, _, test_df = time_split_per_ticker(
        df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )
    print(f"Test set: {len(test_df):,} rows")

    # ── Sample N_EXAMPLES from test set ───────────────────────────────────────
    sample_df = test_df.sample(N_EXAMPLES, random_state=SEED).reset_index(drop=True)
    print(f"Sampled {N_EXAMPLES} test examples for explanation\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    logreg_model = joblib.load("artifacts/models/logreg_blend.joblib")
    rf_model     = joblib.load("artifacts/models/rf_blend.joblib")
    imputer      = joblib.load("artifacts/models/rf_imputer.joblib")
    scaler       = joblib.load("artifacts/models/rf_scaler.joblib")
    print("  Models loaded ✓")

    # ── Load FAISS index ──────────────────────────────────────────────────────
    print("Loading FAISS index...")
    index, metadata = load_index()
    print(f"  Index loaded: {index.ntotal:,} vectors ✓")

    # ── Encode sample articles ────────────────────────────────────────────────
    print("Encoding sample articles...")
    sample_df["text"] = build_text_column(sample_df)
    embeddings = encode_texts(
        sample_df["text"],
        cache_path=CACHE_PATH,
        show_progress=True,
    )
    print(f"  Embeddings: {embeddings.shape} ✓\n")

    # ── Generate explanations ─────────────────────────────────────────────────
    results = []
    separator = "=" * 70

    for i, (_, row) in enumerate(sample_df.iterrows()):
        print(f"{separator}")
        print(f"Example {i+1}/{N_EXAMPLES}")
        print(f"{separator}")
        print(f"Ticker  : {row['ticker']}")
        print(f"Title   : {row['title'][:100]}")
        print(f"Date    : {str(row['published'])[:10]}")
        print(f"Actual  : {'UP ✅' if row['label'] == 1 else 'DOWN ❌'} "
              f"({row['return_1d']*100:+.2f}%)")
        print()

        try:
            result = explain_prediction(
                ticker            = row["ticker"],
                title             = row["title"],
                summary           = row.get("summary", ""),
                article_embedding = embeddings[i],
                index             = index,
                metadata          = metadata,
                logreg_model      = logreg_model,
                rf_model          = rf_model,
                imputer           = imputer,
                scaler            = scaler,
                ohe               = None,        # not needed for RF
                feature_row       = sample_df.iloc[[i]],
                k                 = 5,
                cross_ticker_only = False,
            )

            predicted_correct = (
                (result["direction"] == "UP") == (row["label"] == 1)
            )

            print(f"Predicted: {result['direction']} "
                  f"(blend prob: {result['blend_prob']:.1%}) "
                  f"{'✅ CORRECT' if predicted_correct else '❌ WRONG'}")
            print()
            print("── Retrieved Analogues ──────────────────────────────")
            for j, e in enumerate(result["retrieved_events"], 1):
                ret_str = f"{e['return_1d']*100:+.2f}%"
                dir_str = "UP ✅" if e["label"] == 1 else "DOWN ❌"
                print(f"  {j}. [{e['ticker']}] {e['title'][:80]}")
                print(f"     {e['published']} | {ret_str} | {dir_str} "
                      f"| sim={e['similarity']:.3f}")
            print()
            print("── Explanation ──────────────────────────────────────")
            print(result["explanation"])
            print()

            results.append({
                "example_num"     : i + 1,
                "ticker"          : row["ticker"],
                "title"           : row["title"],
                "published"       : str(row["published"])[:10],
                "actual_label"    : int(row["label"]),
                "actual_return"   : round(float(row["return_1d"]) * 100, 2),
                "predicted_dir"   : result["direction"],
                "blend_prob"      : result["blend_prob"],
                "correct"         : predicted_correct,
                "explanation"     : result["explanation"],
                "n_retrieved"     : len(result["retrieved_events"]),
                "avg_similarity"  : round(
                    sum(e["similarity"] for e in result["retrieved_events"])
                    / max(len(result["retrieved_events"]), 1), 3
                ),
            })

        except Exception as e:
            print(f"  ⚠ Error generating explanation: {e}")
            results.append({
                "example_num": i + 1,
                "ticker": row["ticker"],
                "title": row["title"],
                "error": str(e),
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{separator}")
    print("SUMMARY")
    print(separator)

    valid = [r for r in results if "error" not in r]
    if valid:
        correct_count = sum(1 for r in valid if r["correct"])
        avg_sim = sum(r["avg_similarity"] for r in valid) / len(valid)
        print(f"  Examples run        : {len(valid)}/{N_EXAMPLES}")
        print(f"  Prediction accuracy : {correct_count}/{len(valid)} "
              f"({correct_count/len(valid)*100:.0f}%)")
        print(f"  Avg retrieval sim   : {avg_sim:.3f}")
        print()
        print("  Per-example results:")
        for r in valid:
            tick = "✅" if r["correct"] else "❌"
            print(f"    {r['example_num']:2d}. [{r['ticker']:5s}] "
                  f"{tick} pred={r['predicted_dir']:4s} "
                  f"prob={r['blend_prob']:.2f} "
                  f"sim={r['avg_similarity']:.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    ensure_dir("artifacts/reports")
    out_path = "artifacts/reports/explanations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()