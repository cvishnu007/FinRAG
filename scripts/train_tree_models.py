"""
train_tree_models.py
====================
Trains and evaluates Random Forest and XGBoost models on:
  1. Market features only      (no text)
  2. Market + Sentiment        (no text)
  3. TF-IDF text + Market      (full feature set, XGBoost only)

Saves results to artifacts/reports/metrics_trees.json and prints
a comparison table against the LogReg baseline.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("⚠  XGBoost not installed. Run: pip install xgboost")
    HAS_XGB = False

from src.evaluate import compute_metrics, print_metrics
from src.features import build_feature_frame
from src.market_features import add_market_features
from src.utils import ensure_dir, save_json

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/processed/master_dataset.csv"
PRICES_DIR  = "data/raw/prices"
TIMEZONE    = "America/New_York"
MARKET_OPEN  = "09:30"
MARKET_CLOSE = "16:00"
SEED        = 42
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15

MARKET_COLS = ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]
SENTIMENT_COLS = ["pos_ratio", "neg_ratio", "sent_score"]

LOGREG_BASELINE = {
    "accuracy": 0.5098,
    "precision": 0.5096,
    "recall": 0.6092,
    "f1": 0.5550,
    "roc_auc": 0.5251,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def time_split_per_ticker(df, train_frac=0.70, val_frac=0.15):
    train_parts, val_parts, test_parts = [], [], []
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("published").reset_index(drop=True)
        n = len(group)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])
    return (
        pd.concat(train_parts).sort_values("published").reset_index(drop=True),
        pd.concat(val_parts).sort_values("published").reset_index(drop=True),
        pd.concat(test_parts).sort_values("published").reset_index(drop=True),
    )


def prepare_numeric(train_df, val_df, test_df, feature_cols):
    """Impute + scale numeric features."""
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler  = StandardScaler()

    X_train = imputer.fit_transform(train_df[feature_cols])
    X_val   = imputer.transform(val_df[feature_cols])
    X_test  = imputer.transform(test_df[feature_cols])

    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test


def prepare_tfidf(train_df, val_df, test_df,
                  max_features=5000, min_df=20, max_df=0.5):
    """Fit TF-IDF on train text, transform all splits."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
    )
    X_train = tfidf.fit_transform(train_df["text"].fillna(""))
    X_val   = tfidf.transform(val_df["text"].fillna(""))
    X_test  = tfidf.transform(test_df["text"].fillna(""))
    return X_train, X_val, X_test


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                   model_name):
    results = {}
    for split_name, X, y in [("train", X_train, y_train),
                               ("val",   X_val,   y_val),
                               ("test",  X_test,  y_test)]:
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        m = compute_metrics(y, y_pred, y_proba)
        results[split_name] = m

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print_metrics("Train",      results["train"])
    print_metrics("Validation", results["val"])
    print_metrics("Test",       results["test"])
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)

    # ── Load & prepare data ───────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "label"])
    df = df.sort_values("published").reset_index(drop=True)

    # Dedup
    before = len(df)
    df = df.sort_values(["published", "ticker"])
    df = df.drop_duplicates(subset=["title", "published"], keep="first")
    df = df.sort_values("published").reset_index(drop=True)
    print(f"Cross-ticker dedup: {before:,} → {len(df):,} rows")

    # Add market features
    print("Adding market features...")
    df = add_market_features(
        df,
        prices_dir=PRICES_DIR,
        timezone=TIMEZONE,
        market_close=MARKET_CLOSE,
    )

    # Add text + temporal + sentiment features
    df = build_feature_frame(
        df,
        timezone=TIMEZONE,
        market_open=MARKET_OPEN,
        market_close=MARKET_CLOSE,
        use_sentiment=True,
    )

    # Split
    train_df, val_df, test_df = time_split_per_ticker(
        df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )
    print(f"Split — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    y_train = train_df["label"].astype(int).values
    y_val   = val_df["label"].astype(int).values
    y_test  = test_df["label"].astype(int).values

    all_results = {"logreg_tfidf_baseline": {"test": LOGREG_BASELINE}}

    # ── Experiment 1: Random Forest — market features only ───────────────────
    print("\n\nExperiment 1: Random Forest (market features only)")
    X_train_m, X_val_m, X_test_m = prepare_numeric(
        train_df, val_df, test_df, MARKET_COLS
    )
    rf_market = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_market.fit(X_train_m, y_train)
    all_results["rf_market_only"] = evaluate_model(
        rf_market, X_train_m, y_train, X_val_m, y_val, X_test_m, y_test,
        "Random Forest — market features only"
    )

    # ── Experiment 2: Random Forest — market + sentiment ─────────────────────
    print("\n\nExperiment 2: Random Forest (market + sentiment)")
    X_train_ms, X_val_ms, X_test_ms = prepare_numeric(
        train_df, val_df, test_df, MARKET_COLS + SENTIMENT_COLS
    )
    rf_ms = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf_ms.fit(X_train_ms, y_train)
    all_results["rf_market_sentiment"] = evaluate_model(
        rf_ms, X_train_ms, y_train, X_val_ms, y_val, X_test_ms, y_test,
        "Random Forest — market + sentiment"
    )

    if HAS_XGB:
        # ── Experiment 3: XGBoost — market features only ─────────────────────
        print("\n\nExperiment 3: XGBoost (market features only)")
        xgb_market = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_alpha=1.0,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_market.fit(
            X_train_m, y_train,
            eval_set=[(X_val_m, y_val)],
            verbose=False,
        )
        all_results["xgb_market_only"] = evaluate_model(
            xgb_market, X_train_m, y_train, X_val_m, y_val, X_test_m, y_test,
            "XGBoost — market features only"
        )

        # ── Experiment 4: XGBoost — market + sentiment ────────────────────────
        print("\n\nExperiment 4: XGBoost (market + sentiment)")
        xgb_ms = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_alpha=1.0,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        xgb_ms.fit(
            X_train_ms, y_train,
            eval_set=[(X_val_ms, y_val)],
            verbose=False,
        )
        all_results["xgb_market_sentiment"] = evaluate_model(
            xgb_ms, X_train_ms, y_train, X_val_ms, y_val, X_test_ms, y_test,
            "XGBoost — market + sentiment"
        )

        # ── Experiment 5: XGBoost — TF-IDF + market + sentiment ──────────────
        print("\n\nExperiment 5: XGBoost (TF-IDF + market + sentiment)")
        X_train_t, X_val_t, X_test_t = prepare_tfidf(train_df, val_df, test_df)

        # Combine sparse TF-IDF with dense numeric features
        X_train_full = hstack([X_train_t, csr_matrix(X_train_ms)])
        X_val_full   = hstack([X_val_t,   csr_matrix(X_val_ms)])
        X_test_full  = hstack([X_test_t,  csr_matrix(X_test_ms)])

        xgb_full = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.3,   # lower — many TF-IDF features
            min_child_weight=20,
            reg_alpha=2.0,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",     # fast on sparse
        )
        xgb_full.fit(
            X_train_full, y_train,
            eval_set=[(X_val_full, y_val)],
            verbose=False,
        )
        all_results["xgb_tfidf_market_sentiment"] = evaluate_model(
            xgb_full, X_train_full, y_train,
            X_val_full, y_val, X_test_full, y_test,
            "XGBoost — TF-IDF + market + sentiment"
        )

        # Save best XGBoost model
        ensure_dir("artifacts/models")
        joblib.dump(xgb_full, "artifacts/models/xgb_full_model.joblib")

    # ── Summary comparison table ──────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  COMPARISON TABLE — Test Set")
    print("=" * 65)
    print(f"  {'Model':<40} {'AUC':>6}  {'F1':>6}  {'Acc':>6}")
    print(f"  {'-'*40} {'-'*6}  {'-'*6}  {'-'*6}")

    display_names = {
        "logreg_tfidf_baseline":       "LogReg + TF-IDF (baseline)",
        "rf_market_only":              "Random Forest (market only)",
        "rf_market_sentiment":         "Random Forest (market + sentiment)",
        "xgb_market_only":             "XGBoost (market only)",
        "xgb_market_sentiment":        "XGBoost (market + sentiment)",
        "xgb_tfidf_market_sentiment":  "XGBoost (TF-IDF + market + sentiment)",
    }

    for key, name in display_names.items():
        if key not in all_results:
            continue
        m = all_results[key]["test"]
        print(f"  {name:<40} {m['roc_auc']:>6.4f}  {m['f1']:>6.4f}  {m['accuracy']:>6.4f}")

    print("=" * 65)

    # ── Save all results ──────────────────────────────────────────────────────
    ensure_dir("artifacts/reports")
    save_json("artifacts/reports/metrics_trees.json", all_results)
    print("\nAll results saved to artifacts/reports/metrics_trees.json")


if __name__ == "__main__":
    main()
