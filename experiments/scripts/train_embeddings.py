"""
scripts/train_embeddings.py
============================
Trains Logistic Regression on:
    sentence embeddings + market features + LM sentiment + categorical OHE

Run:
    python -m scripts.train_embeddings
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

from src.embeddings import encode_texts
from src.evaluate import compute_metrics, print_metrics
from src.features import build_feature_frame, build_text_column
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

LABEL_THRESHOLD = 0.002   # same filter as train_blend.py

MARKET_COLS   = ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]
SENTIMENT_COLS = ["pos_ratio", "neg_ratio", "sent_score", "surprise_score"]
CAT_COLS      = ["ticker", "dow", "month", "session", "earnings_season"]

CACHE_PATH    = "artifacts/cache/embeddings.pkl"


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def build_numeric_features(train_df, val_df, test_df, cols):
    """Impute + scale numeric features. Fit on train only."""
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler  = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(train_df[cols]))
    X_val   = scaler.transform(imputer.transform(val_df[cols]))
    X_test  = scaler.transform(imputer.transform(test_df[cols]))

    return X_train, X_val, X_test, imputer, scaler


def build_categorical_features(train_df, val_df, test_df, cols):
    """OneHotEncode categorical columns. Fit on train only."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_train = enc.fit_transform(train_df[cols])
    X_val   = enc.transform(val_df[cols])
    X_test  = enc.transform(test_df[cols])
    return X_train, X_val, X_test, enc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "label"])
    df = df.sort_values("published").reset_index(drop=True)

    # ── Label threshold filter ─────────────────────────────────────────────────
    before = len(df)
    df = df[df["return_1d"].abs() > LABEL_THRESHOLD].copy()
    print(f"Label filter: {before:,} → {len(df):,} rows "
          f"(dropped {before - len(df):,} near-zero rows)")

    # ── Cross-ticker dedup ─────────────────────────────────────────────────────
    before = len(df)
    df = df.sort_values(["published", "ticker"])
    df = df.drop_duplicates(subset=["title", "published"], keep="first")
    df = df.sort_values("published").reset_index(drop=True)
    print(f"Cross-ticker dedup: {before:,} → {len(df):,} rows")

    # ── Market features ────────────────────────────────────────────────────────
    print("Adding market features...")
    df = add_market_features(
        df, prices_dir=PRICES_DIR, timezone=TIMEZONE, market_close=MARKET_CLOSE
    )

    # ── Text / temporal / sentiment features ───────────────────────────────────
    df = build_feature_frame(
        df, timezone=TIMEZONE,
        market_open=MARKET_OPEN, market_close=MARKET_CLOSE,
        use_sentiment=True,
    )

    # ── Split ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = time_split_per_ticker(
        df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )
    print(f"Split — Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    y_train = train_df["label"].astype(int).values
    y_val   = val_df["label"].astype(int).values
    y_test  = test_df["label"].astype(int).values

    # ── Sentence embeddings ────────────────────────────────────────────────────
    # Build text column (title + summary) — same as existing pipeline
    print("\nEncoding sentence embeddings...")
    all_texts = pd.concat([train_df["text"], val_df["text"], test_df["text"]])

    # Encode all at once so cache is populated in one pass
    all_embs = encode_texts(all_texts, cache_path=CACHE_PATH)

    n_train = len(train_df)
    n_val   = len(val_df)
    emb_train = all_embs[:n_train]
    emb_val   = all_embs[n_train : n_train + n_val]
    emb_test  = all_embs[n_train + n_val :]

    print(f"  Embedding shape — train: {emb_train.shape}, "
          f"val: {emb_val.shape}, test: {emb_test.shape}")

    # ── Numeric features ───────────────────────────────────────────────────────
    print("Building numeric features...")
    num_cols = MARKET_COLS + SENTIMENT_COLS
    X_num_train, X_num_val, X_num_test, imputer, scaler = build_numeric_features(
        train_df, val_df, test_df, num_cols
    )

    # ── Categorical features ───────────────────────────────────────────────────
    print("Building categorical features...")
    X_cat_train, X_cat_val, X_cat_test, ohe = build_categorical_features(
        train_df, val_df, test_df, CAT_COLS
    )

    # ── Combine all features ───────────────────────────────────────────────────
    # Stack: [384-dim embedding | 9-dim numeric | ~35-dim OHE]
    X_train = hstack([csr_matrix(emb_train), csr_matrix(X_num_train), X_cat_train])
    X_val   = hstack([csr_matrix(emb_val),   csr_matrix(X_num_val),   X_cat_val])
    X_test  = hstack([csr_matrix(emb_test),  csr_matrix(X_num_test),  X_cat_test])

    print(f"  Final feature matrix — train: {X_train.shape}")

    # ── Grid search over C ─────────────────────────────────────────────────────
    print("\nTraining Logistic Regression (grid search over C)...")
    best_model  = None
    best_auc    = -1.0
    best_c      = None

    for c_val in [0.01, 0.05, 0.1, 0.3, 1.0]:
        clf = LogisticRegression(
            penalty="l2",
            C=c_val,
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        val_proba = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        print(f"  C={c_val:<5}  val AUC={auc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_model = clf
            best_c     = c_val

    print(f"\n  Best C={best_c}  val AUC={best_auc:.4f}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    results = {}
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        y_pred  = best_model.predict(X)
        y_proba = best_model.predict_proba(X)[:, 1]
        m = compute_metrics(y, y_pred, y_proba)
        results[split_name] = m
        print_metrics(split_name.capitalize(), m)

    # ── Comparison ─────────────────────────────────────────────────────────────
    BASELINES = {
        "LogReg + TF-IDF":               0.5251,
        "Blend + LM dict + filter":      0.5432,
    }
    print("\n" + "=" * 55)
    print("  AUC COMPARISON — Test Set")
    print("=" * 55)
    for name, auc in BASELINES.items():
        print(f"  {name:<40} {auc:.4f}")
    print(f"  {'MiniLM embeddings (this run)':<40} {results['test']['roc_auc']:.4f}")
    delta = results["test"]["roc_auc"] - 0.5432
    print(f"  {'Delta vs blend baseline':<40} {delta:+.4f}")
    print("=" * 55)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    ensure_dir("artifacts/models")
    ensure_dir("artifacts/reports")

    joblib.dump(best_model, "artifacts/models/minilm_logreg.joblib")
    joblib.dump(imputer,    "artifacts/models/minilm_imputer.joblib")
    joblib.dump(scaler,     "artifacts/models/minilm_scaler.joblib")
    joblib.dump(ohe,        "artifacts/models/minilm_ohe.joblib")

    meta = {
        "model_name":     "all-MiniLM-L6-v2",
        "best_c":         best_c,
        "label_threshold": LABEL_THRESHOLD,
        "num_cols":       num_cols,
        "cat_cols":       CAT_COLS,
        "embedding_dim":  384,
        "cache_path":     CACHE_PATH,
    }
    joblib.dump(meta, "artifacts/models/minilm_meta.joblib")
    save_json("artifacts/reports/metrics_minilm.json", {
        "best_c": best_c,
        **results,
    })

    print("\nArtifacts saved:")
    print("  artifacts/models/minilm_logreg.joblib")
    print("  artifacts/models/minilm_meta.joblib")
    print("  artifacts/reports/metrics_minilm.json")


if __name__ == "__main__":
    main()