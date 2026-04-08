"""
train_blend.py
==============
Trains LogReg + Random Forest on a threshold-filtered dataset
(drops near-zero returns) and blends their predictions.

Key improvements over baseline:
  1. Label threshold filter  — drops rows where |return| < THRESHOLD
     so the model only trains on "clear" moves, not noise.
  2. Loughran-McDonald sentiment (via updated features.py)
  3. Blended prediction      — weighted average of LogReg + RF probas.
  4. Optimal threshold tuning on val set instead of hardcoded 0.5.

Run:
    python -m scripts.train_blend
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

from src.evaluate import compute_metrics, print_metrics
from src.features import build_feature_frame
from src.market_features import add_market_features
from src.utils import ensure_dir, save_json

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/processed/master_dataset.csv"
PRICES_DIR   = "data/raw/prices"
TIMEZONE     = "America/New_York"
MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"
SEED         = 42
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15

# Label threshold: drop rows where |return| < this value.
# 0.002 = 0.2% — removes the noisy near-zero middle.
# Try 0.001, 0.002, 0.003 to see which gives best val AUC.
LABEL_THRESHOLD = 0.002

MARKET_COLS   = ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]
SENTIMENT_COLS = ["pos_ratio", "neg_ratio", "sent_score", "surprise_score"]

# Blend weight for LogReg vs RF (must sum to 1.0)
# LogReg is better at text signal; RF is better at market feature interactions
LOGREG_WEIGHT = 0.60
RF_WEIGHT     = 0.40


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_first_col(x):
    return x.iloc[:, 0]


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


def find_best_threshold(y_true, y_proba):
    """Find the probability cutoff that maximises F1 on val set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = np.where(
        (precisions + recalls) == 0,
        0,
        2 * precisions * recalls / (precisions + recalls),
    )
    best_idx = f1s.argmax()
    # thresholds has one fewer element than precisions/recalls
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    return float(thresholds[best_idx])


def apply_threshold(y_proba, threshold):
    return (y_proba >= threshold).astype(int)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "label"])
    df = df.sort_values("published").reset_index(drop=True)

    # ── Label threshold filter ────────────────────────────────────────────────
    # "excess_return" if you applied market-adjusted returns in merge_data.py,
    # otherwise falls back to "return_1d"
    return_col = "excess_return" if "excess_return" in df.columns else "return_1d"

    before = len(df)
    if return_col in df.columns:
        df = df[df[return_col].abs() > LABEL_THRESHOLD].copy()
        print(
            f"Label threshold filter ({return_col} > ±{LABEL_THRESHOLD*100:.1f}%): "
            f"{before:,} → {len(df):,} rows "
            f"(dropped {before - len(df):,} near-zero rows, "
            f"{(before - len(df)) / before * 100:.1f}%)"
        )
    else:
        print(
            f"⚠  Column '{return_col}' not found — skipping threshold filter. "
            f"Run merge_data.py first."
        )

    # ── Cross-ticker dedup ────────────────────────────────────────────────────
    before = len(df)
    df = df.sort_values(["published", "ticker"])
    df = df.drop_duplicates(subset=["title", "published"], keep="first")
    df = df.sort_values("published").reset_index(drop=True)
    print(f"Cross-ticker dedup: {before:,} → {len(df):,} rows")

    # ── Market features ───────────────────────────────────────────────────────
    print("Adding market features...")
    df = add_market_features(
        df,
        prices_dir=PRICES_DIR,
        timezone=TIMEZONE,
        market_close=MARKET_CLOSE,
    )

    # ── Text / temporal / sentiment features ──────────────────────────────────
    df = build_feature_frame(
        df,
        timezone=TIMEZONE,
        market_open=MARKET_OPEN,
        market_close=MARKET_CLOSE,
        use_sentiment=True,
    )

    # ── Per-ticker chronological split ────────────────────────────────────────
    train_df, val_df, test_df = time_split_per_ticker(
        df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )
    print(
        f"Split — Train: {len(train_df):,}  "
        f"Val: {len(val_df):,}  "
        f"Test: {len(test_df):,}"
    )

    y_train = train_df["label"].astype(int).values
    y_val   = val_df["label"].astype(int).values
    y_test  = test_df["label"].astype(int).values

    print(f"\nLabel balance after filtering:")
    print(f"  Train UP%: {y_train.mean()*100:.1f}%")
    print(f"  Val   UP%: {y_val.mean()*100:.1f}%")
    print(f"  Test  UP%: {y_test.mean()*100:.1f}%")

    # ── Feature spec ──────────────────────────────────────────────────────────
    text_col  = "text"
    cat_cols  = ["ticker", "dow", "month", "session", "earnings_season"]
    num_cols  = SENTIMENT_COLS + MARKET_COLS

    # ── Build sklearn preprocessor ────────────────────────────────────────────
    text_transformer = Pipeline([
        ("to_text", FunctionTransformer(_extract_first_col, validate=False)),
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),   # bigrams — catches "beat estimates", "miss expectations"
            min_df=10,
            max_df=0.5,
            stop_words="english",
        )),
    ])

    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("text", text_transformer, [text_col]),
        ("cat",  cat_transformer,  cat_cols),
        ("num",  num_transformer,  num_cols),
    ])

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    print("\n── Training Logistic Regression ──")
    best_logreg = None
    best_logreg_auc = -1.0

    for c_val in [0.001, 0.003, 0.01, 0.03]:
        clf = LogisticRegression(
            penalty="l2",
            C=c_val,
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            random_state=SEED,
        )
        pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
        pipe.fit(train_df, y_train)

        val_proba = pipe.predict_proba(val_df)[:, 1]
        val_metrics = compute_metrics(
            y_val, (val_proba >= 0.5).astype(int), val_proba
        )
        print(f"  C={c_val:.3f}  val AUC={val_metrics['roc_auc']:.4f}")

        if val_metrics["roc_auc"] > best_logreg_auc:
            best_logreg_auc = val_metrics["roc_auc"]
            best_logreg = pipe

    print(f"  → Best val AUC: {best_logreg_auc:.4f}")

    # ── Model 2: Random Forest (market + sentiment only — no TF-IDF) ──────────
    print("\n── Training Random Forest ──")
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler  = StandardScaler()

    X_train_rf = imputer.fit_transform(train_df[MARKET_COLS + SENTIMENT_COLS])
    X_val_rf   = imputer.transform(val_df[MARKET_COLS + SENTIMENT_COLS])
    X_test_rf  = imputer.transform(test_df[MARKET_COLS + SENTIMENT_COLS])

    X_train_rf = scaler.fit_transform(X_train_rf)
    X_val_rf   = scaler.transform(X_val_rf)
    X_test_rf  = scaler.transform(X_test_rf)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,           # tighter than before to reduce overfit
        min_samples_leaf=40,   # increased — forces more conservative splits
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_rf, y_train)

    rf_val_proba = rf.predict_proba(X_val_rf)[:, 1]
    rf_val_metrics = compute_metrics(
        y_val, (rf_val_proba >= 0.5).astype(int), rf_val_proba
    )
    print(f"  RF val AUC: {rf_val_metrics['roc_auc']:.4f}")

    # ── Blend probabilities ───────────────────────────────────────────────────
    print(f"\n── Blending ({LOGREG_WEIGHT:.0%} LogReg + {RF_WEIGHT:.0%} RF) ──")

    logreg_val_proba = best_logreg.predict_proba(val_df)[:, 1]
    blend_val_proba  = LOGREG_WEIGHT * logreg_val_proba + RF_WEIGHT * rf_val_proba

    # Find optimal decision threshold on validation set
    best_thresh = find_best_threshold(y_val, blend_val_proba)
    print(f"  Optimal val threshold: {best_thresh:.3f}  (default was 0.500)")

    # ── Evaluate all variants ─────────────────────────────────────────────────
    logreg_test_proba = best_logreg.predict_proba(test_df)[:, 1]
    rf_test_proba     = rf.predict_proba(X_test_rf)[:, 1]
    blend_test_proba  = LOGREG_WEIGHT * logreg_test_proba + RF_WEIGHT * rf_test_proba

    print("\n" + "=" * 55)
    print("  RESULTS")
    print("=" * 55)

    results = {}

    # LogReg alone (default threshold)
    logreg_test_pred = (logreg_test_proba >= 0.5).astype(int)
    m = compute_metrics(y_test, logreg_test_pred, logreg_test_proba)
    results["logreg_default_threshold"] = m
    print_metrics("LogReg (threshold=0.5)", m)

    # Blend (default threshold)
    blend_pred_default = (blend_test_proba >= 0.5).astype(int)
    m = compute_metrics(y_test, blend_pred_default, blend_test_proba)
    results["blend_default_threshold"] = m
    print_metrics("Blend (threshold=0.5)", m)

    # Blend (optimal threshold from val)
    blend_pred_optimal = apply_threshold(blend_test_proba, best_thresh)
    m = compute_metrics(y_test, blend_pred_optimal, blend_test_proba)
    results["blend_optimal_threshold"] = m
    print_metrics(f"Blend (threshold={best_thresh:.3f}, tuned on val)", m)

    # ── Comparison table ──────────────────────────────────────────────────────
    BASELINE = {"roc_auc": 0.5251, "f1": 0.5550, "accuracy": 0.5098}

    print("\n" + "=" * 65)
    print("  COMPARISON TABLE — Test Set")
    print("=" * 65)
    print(f"  {'Model':<42} {'AUC':>6}  {'F1':>6}  {'Acc':>6}")
    print(f"  {'-'*42} {'-'*6}  {'-'*6}  {'-'*6}")

    rows = [
        ("LogReg baseline (no filter)",        BASELINE),
        ("LogReg + LM sentiment + filter",     results["logreg_default_threshold"]),
        ("Blend default threshold",             results["blend_default_threshold"]),
        ("Blend optimal threshold (val-tuned)", results["blend_optimal_threshold"]),
    ]
    for name, m in rows:
        print(
            f"  {name:<42} "
            f"{m['roc_auc']:>6.4f}  "
            f"{m['f1']:>6.4f}  "
            f"{m['accuracy']:>6.4f}"
        )
    print("=" * 65)

    # ── Save ──────────────────────────────────────────────────────────────────
    ensure_dir("artifacts/models")
    ensure_dir("artifacts/reports")

    joblib.dump(best_logreg,            "artifacts/models/logreg_blend.joblib")
    joblib.dump(rf,                     "artifacts/models/rf_blend.joblib")
    joblib.dump(imputer,                "artifacts/models/rf_imputer.joblib")
    joblib.dump(scaler,                 "artifacts/models/rf_scaler.joblib")
    joblib.dump({"threshold": best_thresh,
                 "logreg_weight": LOGREG_WEIGHT,
                 "rf_weight": RF_WEIGHT},
                "artifacts/models/blend_meta.joblib")

    save_json("artifacts/reports/metrics_blend.json", {
        "threshold_used": best_thresh,
        "label_threshold": LABEL_THRESHOLD,
        "return_col": return_col,
        **{k: v for k, v in results.items()},
    })

    print("\nArtifacts saved:")
    print("  artifacts/models/logreg_blend.joblib")
    print("  artifacts/models/rf_blend.joblib")
    print("  artifacts/models/blend_meta.joblib")
    print("  artifacts/reports/metrics_blend.json")


if __name__ == "__main__":
    main()
