"""
scripts/train_finbert.py
========================
Trains LogReg on FinBERT embeddings + market + sentiment features.
Directly comparable to train_embeddings.py (MiniLM baseline).

Run:
    python -m scripts.train_finbert
"""

import os
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
from src.utils import ensure_dir, save_json, load_yaml_config

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
SEED         = 42

# ── Switch model here ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "yiyanghkust/finbert-tone"
# EMBEDDING_MODEL = "ProsusAI/finbert"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # re-run baseline for fair comparison

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


def main():
    np.random.seed(SEED)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("=" * 55)

    # ── Load and prepare ──────────────────────────────────────────────────────
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

    train_df, val_df, test_df = time_split_per_ticker(
        df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )
    print(f"Split — Train: {len(train_df):,}  "
          f"Val: {len(val_df):,}  Test: {len(test_df):,}")

    y_train = train_df["label"].astype(int).values
    y_val   = val_df["label"].astype(int).values
    y_test  = test_df["label"].astype(int).values

    # ── Encode with FinBERT ───────────────────────────────────────────────────
    print(f"\nEncoding with {EMBEDDING_MODEL}...")
    all_texts = pd.concat([
        train_df["text"], val_df["text"], test_df["text"]
    ]).reset_index(drop=True)

    all_embs = encode_texts(
        all_texts,
        cache_path=CACHE_PATH,
        model_name=EMBEDDING_MODEL,
        batch_size=16,      # smaller for 768-dim model
        show_progress=True,
    )

    n_train = len(train_df)
    n_val   = len(val_df)
    emb_train = all_embs[:n_train]
    emb_val   = all_embs[n_train : n_train + n_val]
    emb_test  = all_embs[n_train + n_val :]

    print(f"  Embedding shape — train: {emb_train.shape}, "
          f"val: {emb_val.shape}, test: {emb_test.shape}")

    # ── Numeric features ──────────────────────────────────────────────────────
    num_cols = MARKET_COLS + SENTIMENT_COLS
    imputer  = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler   = StandardScaler()

    X_num_train = scaler.fit_transform(imputer.fit_transform(train_df[num_cols]))
    X_num_val   = scaler.transform(imputer.transform(val_df[num_cols]))
    X_num_test  = scaler.transform(imputer.transform(test_df[num_cols]))

    # ── Categorical features ──────────────────────────────────────────────────
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat_train = ohe.fit_transform(train_df[CAT_COLS])
    X_cat_val   = ohe.transform(val_df[CAT_COLS])
    X_cat_test  = ohe.transform(test_df[CAT_COLS])

    # ── Combine ───────────────────────────────────────────────────────────────
    X_train = hstack([csr_matrix(emb_train), csr_matrix(X_num_train), X_cat_train])
    X_val   = hstack([csr_matrix(emb_val),   csr_matrix(X_num_val),   X_cat_val])
    X_test  = hstack([csr_matrix(emb_test),  csr_matrix(X_num_test),  X_cat_test])

    print(f"  Final feature matrix: {X_train.shape}")

    # ── Grid search ───────────────────────────────────────────────────────────
    print("\nTraining Logistic Regression...")
    best_model = None
    best_auc   = -1.0
    best_c     = None

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

    # ── Final evaluation ──────────────────────────────────────────────────────
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

    # ── Comparison table ──────────────────────────────────────────────────────
    BASELINES = {
        "LogReg + TF-IDF"           : 0.5251,
        "Blend + LM dict + filter"  : 0.5432,
        "MiniLM embeddings"         : 0.5250,
    }

    print("\n" + "=" * 58)
    print("  AUC COMPARISON — Test Set")
    print("=" * 58)
    for name, auc in BASELINES.items():
        print(f"  {name:<42} {auc:.4f}")
    print(f"  {EMBEDDING_MODEL:<42} "
          f"{results['test']['roc_auc']:.4f}")
    delta = results["test"]["roc_auc"] - 0.5432
    print(f"  {'Delta vs blend baseline':<42} {delta:+.4f}")
    print("=" * 58)

    # ── Save ──────────────────────────────────────────────────────────────────
    ensure_dir("artifacts/models")
    ensure_dir("artifacts/reports")

    model_name_safe = EMBEDDING_MODEL.replace("/", "_")
    joblib.dump(best_model, f"artifacts/models/{model_name_safe}_logreg.joblib")

    save_json(f"artifacts/reports/metrics_{model_name_safe}.json", {
        "embedding_model" : EMBEDDING_MODEL,
        "best_c"          : best_c,
        "label_threshold" : LABEL_THRESHOLD,
        **results,
    })

    print(f"\nArtifacts saved:")
    print(f"  artifacts/models/{model_name_safe}_logreg.joblib")
    print(f"  artifacts/reports/metrics_{model_name_safe}.json")


if __name__ == "__main__":
    main()