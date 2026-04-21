"""
scripts/evaluate_system.py
===========================
Full system evaluation on the held-out test set.
Measures prediction accuracy, retrieval quality, and explanation
grounding across 100 random test examples.

Run:
    set GROQ_API_KEY=gsk_your_key_here
    python -m scripts.evaluate_system
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from src.embeddings import encode_texts
from src.retrieval import load_index
from src.explain import explain_prediction, call_llm
from src.features import build_feature_frame, build_text_column
from src.market_features import add_market_features
from src.utils import ensure_dir
from scripts.run_explain import _extract_first_col

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
SEED            = 0

MARKET_COLS    = ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]
SENTIMENT_COLS = ["pos_ratio", "neg_ratio", "sent_score", "surprise_score"]


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


def score_grounding(explanation: str, retrieved_events: list) -> dict:
    """
    Simple grounding check — counts how many retrieved event headlines
    are referenced (even partially) in the explanation text.
    Not perfect but objective and fast.
    """
    explanation_lower = explanation.lower()
    grounded = 0

    for e in retrieved_events:
        # Check if key words from the title appear in the explanation
        title_words = [
            w for w in e["title"].lower().split()
            if len(w) > 5 and w.isalpha()
        ]
        # If at least 2 significant words from the title appear → grounded
        matches = sum(1 for w in title_words if w in explanation_lower)
        if matches >= 2:
            grounded += 1

    # Also check if the date appears
    dates_cited = sum(
        1 for e in retrieved_events
        if e["published"][:7] in explanation   # YYYY-MM
    )

    return {
        "events_referenced" : grounded,
        "dates_cited"       : dates_cited,
        "grounding_score"   : grounded / max(len(retrieved_events), 1),
    }


def main():
    # ── Load and prepare data ─────────────────────────────────────────────────
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

    # Sample N_EXAMPLES
    sample_df = test_df.sample(
        N_EXAMPLES, random_state=SEED
    ).reset_index(drop=True)

    # ── Load models and index ─────────────────────────────────────────────────
    print("\nLoading models...")
    logreg_model = joblib.load("artifacts/models/logreg_blend.joblib")
    rf_model     = joblib.load("artifacts/models/rf_blend.joblib")
    imputer      = joblib.load("artifacts/models/rf_imputer.joblib")
    scaler       = joblib.load("artifacts/models/rf_scaler.joblib")

    print("Loading FAISS index...")
    index, metadata = load_index()
    print(f"  {index.ntotal:,} vectors ✓")

    # ── Encode all sample articles ────────────────────────────────────────────
    print("Encoding articles...")
    sample_df["text"] = build_text_column(sample_df)
    embeddings = encode_texts(
        sample_df["text"], cache_path=CACHE_PATH, show_progress=True
    )

    # ── Run evaluation loop ───────────────────────────────────────────────────
    print(f"\nRunning evaluation on {N_EXAMPLES} examples...\n")

    results      = []
    errors       = 0
    ticker_stats = {}

    for i, (_, row) in enumerate(sample_df.iterrows()):
        print(f"  [{i+1:3d}/{N_EXAMPLES}] {row['ticker']:6s} | "
              f"{str(row['published'])[:10]} | "
              f"{row['title'][:55]:<55}", end=" ... ", flush=True)

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
                ohe               = None,
                feature_row       = sample_df.iloc[[i]],
                k                 = 5,
            )

            actual_label = int(row["label"])
            predicted_up = result["direction"] == "UP"
            correct      = (predicted_up == (actual_label == 1))

            grounding = score_grounding(
                result["explanation"],
                result["retrieved_events"]
            )

            avg_sim = (
                sum(e["similarity"] for e in result["retrieved_events"])
                / max(len(result["retrieved_events"]), 1)
            )

            tick = "✅" if correct else "❌"
            print(f"{tick} prob={result['blend_prob']:.2f} "
                  f"sim={avg_sim:.3f} "
                  f"grnd={grounding['grounding_score']:.2f}")

# After explain_prediction returns result:

            verification_passed = result.get("verified", False)
            used_template       = result.get("used_template", False)
            attempts            = result.get("attempts", 1)
            verif_errors        = result.get("verification_errors", [])

            record = {
    "example_num"         : i + 1,
    "ticker"              : row["ticker"],
    "title"               : row["title"],
    "published"           : str(row["published"])[:10],
    "actual_label"        : actual_label,
    "actual_return_pct"   : round(float(row["return_1d"]) * 100, 3),
    "predicted_dir"       : result["direction"],
    "blend_prob"          : result["blend_prob"],
    "correct"             : correct,
    "avg_similarity"      : round(avg_sim, 4),
    "n_retrieved"         : len(result["retrieved_events"]),
    # grounding fields — these were missing
    "grounding_score"     : grounding["grounding_score"],
    "events_referenced"   : grounding["events_referenced"],
    "dates_cited"         : grounding["dates_cited"],
    # verification fields
    "verified"            : verification_passed,
    "used_template"       : used_template,
    "llm_attempts"        : attempts,
    "verification_errors" : verif_errors,
    "explanation"         : result["explanation"],
}
            results.append(record)
            # Per-ticker tracking
            t = row["ticker"]
            if t not in ticker_stats:
                ticker_stats[t] = {"correct": 0, "total": 0}
            ticker_stats[t]["total"]  += 1
            ticker_stats[t]["correct"] += int(correct)

        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1
            results.append({
                "example_num": i + 1,
                "ticker": row["ticker"],
                "error": str(e),
            })

    # ── Compute aggregate metrics ─────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    n     = len(valid)

    if n == 0:
        print("No valid results.")
        return

    accuracy        = sum(r["correct"] for r in valid) / n
    avg_sim         = sum(r["avg_similarity"] for r in valid) / n
    avg_grounding   = sum(r["grounding_score"] for r in valid) / n
    avg_prob_up     = sum(r["blend_prob"] for r in valid if r["predicted_dir"] == "UP") / max(sum(1 for r in valid if r["predicted_dir"] == "UP"), 1)
    avg_prob_down   = sum(r["blend_prob"] for r in valid if r["predicted_dir"] == "DOWN") / max(sum(1 for r in valid if r["predicted_dir"] == "DOWN"), 1)

    # Accuracy by return magnitude bucket
    buckets = {
        "small  (|ret| < 1%)": [],
        "medium (1-3%)":        [],
        "large  (|ret| > 3%)":  [],
    }
    for r in valid:
        abs_ret = abs(r["actual_return_pct"])
        if abs_ret < 1.0:
            buckets["small  (|ret| < 1%)"].append(r["correct"])
        elif abs_ret < 3.0:
            buckets["medium (1-3%)"].append(r["correct"])
        else:
            buckets["large  (|ret| > 3%)"].append(r["correct"])

    # Accuracy by similarity bucket
    high_sim = [r for r in valid if r["avg_similarity"] >= 0.5]
    low_sim  = [r for r in valid if r["avg_similarity"] < 0.5]

    # ── Print final report ────────────────────────────────────────────────────
    sep = "=" * 65

    print(f"\n\n{sep}")
    print("  FULL SYSTEM EVALUATION REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)

    print(f"\n── Dataset ──────────────────────────────────────────────────")
    print(f"  Test set size    : {len(test_df):,} rows")
    print(f"  Sample evaluated : {n} examples ({errors} errors)")
    print(f"  Tickers covered  : {len(ticker_stats)}")

    print(f"\n── Prediction Performance ───────────────────────────────────")
    print(f"  Overall accuracy     : {accuracy*100:.1f}%  ({sum(r['correct'] for r in valid)}/{n})")
    print(f"  Baseline model AUC   : 0.5432  (blend, full test set)")
    print(f"  Avg blend prob (UP)  : {avg_prob_up:.3f}")
    print(f"  Avg blend prob (DOWN): {avg_prob_down:.3f}")

    print(f"\n── Accuracy by Return Magnitude ─────────────────────────────")
    for bucket, vals in buckets.items():
        if vals:
            acc = sum(vals) / len(vals)
            print(f"  {bucket}: {acc*100:.1f}%  (n={len(vals)})")

    print(f"\n── Retrieval Quality ────────────────────────────────────────")
    print(f"  Avg cosine similarity    : {avg_sim:.3f}")
    print(f"  High-sim (>=0.5) count  : {len(high_sim)}/{n}")
    if high_sim:
        acc_high = sum(r["correct"] for r in high_sim) / len(high_sim)
        print(f"  Accuracy when sim>=0.5  : {acc_high*100:.1f}%")
    if low_sim:
        acc_low = sum(r["correct"] for r in low_sim) / len(low_sim)
        print(f"  Accuracy when sim<0.5   : {acc_low*100:.1f}%")

    print(f"\n── Explanation Grounding ────────────────────────────────────")
    print(f"  Avg grounding score      : {avg_grounding:.3f}  "
          f"(fraction of retrieved events cited)")
    grounded_fully = sum(1 for r in valid if r["grounding_score"] >= 0.6)
    print(f"  Well-grounded (>=60%)    : {grounded_fully}/{n} "
          f"({grounded_fully/n*100:.0f}%)")
    avg_dates = sum(r["dates_cited"] for r in valid) / n
    print(f"  Avg dates cited          : {avg_dates:.1f}/5")

    print(f"\n── Per-Ticker Accuracy ──────────────────────────────────────")
    for ticker, stats in sorted(ticker_stats.items()):
        acc_t = stats["correct"] / stats["total"]
        bar   = "█" * stats["correct"] + "░" * (stats["total"] - stats["correct"])
        print(f"  {ticker:6s}: {acc_t*100:5.1f}%  [{bar}]  "
              f"({stats['correct']}/{stats['total']})")
    print(f"\n── Explanation Verification ─────────────────────────────────")
    verified_count  = sum(1 for r in valid if r.get("verified", False))
    template_count  = sum(1 for r in valid if r.get("used_template", False))
    one_shot_count  = sum(1 for r in valid if r.get("verified") and r.get("llm_attempts") == 1)
    retry_count     = sum(1 for r in valid if r.get("verified") and r.get("llm_attempts", 1) > 1)

    print(f"  Verified (factually correct)     : {verified_count}/{n} ({verified_count/n*100:.0f}%)")
    print(f"  Passed first attempt             : {one_shot_count}/{n}")
    print(f"  Passed after retry               : {retry_count}/{n}")
    print(f"  Used template fallback           : {template_count}/{n}")

    # Most common errors
    all_errors = []
    for r in valid:
        all_errors.extend(r.get("verification_errors", []))

    if all_errors:
        print(f"\n  Most common verification failures:")
        from collections import Counter
        # Categorize errors by their prefix
        error_types = Counter()
        for e in all_errors:
            if "Direction mismatch" in e:
                error_types["Direction mismatch"] += 1
            elif "Confidence mismatch" in e:
                error_types["Confidence mismatch"] += 1
            elif "Hallucinated event" in e:
                error_types["Hallucinated event"] += 1
            elif "Direction error for" in e:
                error_types["Wrong event direction"] += 1
            elif "Return error for" in e:
                error_types["Wrong return value"] += 1
            else:
                error_types["Other"] += 1
        for error_type, count in error_types.most_common():
            print(f"    {error_type:<30}: {count}")

    print(f"\n{sep}")

    # ── Save ──────────────────────────────────────────────────────────────────
    ensure_dir("artifacts/reports")

    summary = {
        "generated_at"      : datetime.now().isoformat(),
        "n_examples"        : n,
        "errors"            : errors,
        "accuracy"          : round(accuracy, 4),
        "baseline_auc"      : 0.5432,
        "avg_similarity"    : round(avg_sim, 4),
        "avg_grounding"     : round(avg_grounding, 4),
        "well_grounded_pct" : round(grounded_fully / n, 4),
        "ticker_stats"      : ticker_stats,
        "bucket_accuracy"   : {
            k: round(sum(v)/len(v), 4) if v else None
            for k, v in buckets.items()
        },
        "high_sim_accuracy" : round(sum(r["correct"] for r in high_sim) / max(len(high_sim), 1), 4),
        "low_sim_accuracy"  : round(sum(r["correct"] for r in low_sim)  / max(len(low_sim),  1), 4),
    }

    with open("artifacts/reports/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("artifacts/reports/evaluation_details.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Summary → artifacts/reports/evaluation_summary.json")
    print(f"  Details → artifacts/reports/evaluation_details.json")
    print(f"\n✅ Evaluation complete. Project is done.")


if __name__ == "__main__":
    main()