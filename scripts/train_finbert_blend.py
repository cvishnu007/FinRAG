"""
scripts/train_finbert_blend.py
==============================
Combines FinBERT-tone embeddings WITH the full blend pipeline:
  - FinBERT-tone 768-dim embeddings (cached, instant)
  - Loughran-McDonald sentiment features
  - Market momentum features
  - Label threshold filter (±0.2%)
  - 60% LogReg + 40% RF blend

This is the one experiment not yet run — FinBERT + all other improvements.

Run:
    python -m scripts.train_finbert_blend
"""

import os
import numpy as np
import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve

from src.embeddings import encode_texts
from src.evaluate import compute_metrics, print_metrics
from src.features import build_feature_frame, build_text_column
from src.market_features import add_market_features
from src.utils import ensure_dir, save_json

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
SEED            = 42

EMBEDDING_MODEL = "yiyanghkust/finbert-tone"   # already cached — instant
LOGREG_WEIGHT   = 0.60
RF_WEIGHT       = 0.40

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

    print("=" * 58)
    print("  FinBERT-tone + Full Blend Pipeline")
    print("=" * 58)

    # ── Load and prepare ──────────────────────────────────────────────────────
    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "label"])
    df = df.sort_values("published").reset_index(drop=True)

    before = len(df)
    df = df[df["return_1d"].abs() > LABEL_THRESHOLD].copy()
    print(f"Label filter: {before:,} → {len(df):,} rows "
          f"(dropped {before - len(df):,} near-zero rows)")

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

    print(f"\nLabel balance:")
    print(f"  Train UP%: {y_train.mean()*100:.1f}%")
    print(f"  Val   UP%: {y_val.mean()*100:.1f}%")
    print(f"  Test  UP%: {y_test.mean()*100:.1f}%")

    # ── FinBERT embeddings (served from cache) ────────────────────────────────
    print(f"\nLoading FinBERT embeddings from cache...")
    all_texts = pd.concat([
        train_df["text"], val_df["text"], test_df["text"]
    ]).reset_index(drop=True)

    all_embs = encode_texts(
        all_texts,
        cache_path=CACHE_PATH,
        model_name=EMBEDDING_MODEL,
        batch_size=16,
        show_progress=True,
    )

    n_train   = len(train_df)
    n_val     = len(val_df)
    emb_train = all_embs[:n_train]
    emb_val   = all_embs[n_train : n_train + n_val]
    emb_test  = all_embs[n_train + n_val :]
    print(f"  Embedding shape: {emb_train.shape} ✓")

    # ── Numeric features (market + LM sentiment) ──────────────────────────────
    print("\nBuilding numeric features...")
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

    # ── Full feature matrix for LogReg ────────────────────────────────────────
    # [768-dim FinBERT | 9-dim numeric | ~35-dim OHE] = ~812 features
    X_train_full = hstack([
        csr_matrix(emb_train), csr_matrix(X_num_train), X_cat_train
    ])
    X_val_full   = hstack([
        csr_matrix(emb_val),   csr_matrix(X_num_val),   X_cat_val
    ])
    X_test_full  = hstack([
        csr_matrix(emb_test),  csr_matrix(X_num_test),  X_cat_test
    ])
    print(f"  Full feature matrix: {X_train_full.shape}")

    # ── RF features (market + sentiment only, no embeddings) ─────────────────
    # RF doesn't benefit from 768-dim sparse embeddings — keep it lean
    rf_cols = MARKET_COLS + SENTIMENT_COLS
    imputer_rf = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler_rf  = StandardScaler()

    X_rf_train = scaler_rf.fit_transform(imputer_rf.fit_transform(train_df[rf_cols]))
    X_rf_val   = scaler_rf.transform(imputer_rf.transform(val_df[rf_cols]))
    X_rf_test  = scaler_rf.transform(imputer_rf.transform(test_df[rf_cols]))

    # ── Model 1: Logistic Regression (FinBERT + all features) ────────────────
    print("\n── Training Logistic Regression (FinBERT + market + sentiment) ──")
    best_logreg     = None
    best_logreg_auc = -1.0
    best_c          = None

    for c_val in [0.001, 0.003, 0.01, 0.03, 0.1]:
        clf = LogisticRegression(
            penalty="l2",
            C=c_val,
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(X_train_full, y_train)
        val_proba = clf.predict_proba(X_val_full)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        print(f"  C={c_val:<6}  val AUC={auc:.4f}")

        if auc > best_logreg_auc:
            best_logreg_auc = auc
            best_logreg     = clf
            best_c          = c_val

    print(f"  → Best C={best_c}  val AUC={best_logreg_auc:.4f}")

    # ── Model 2: Random Forest (market + sentiment only) ──────────────────────
    print("\n── Training Random Forest (market + sentiment) ──")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=40,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(X_rf_train, y_train)
    rf_val_proba = rf.predict_proba(X_rf_val)[:, 1]
    rf_val_auc   = roc_auc_score(y_val, rf_val_proba)
    print(f"  RF val AUC: {rf_val_auc:.4f}")

    # ── Blend ─────────────────────────────────────────────────────────────────
    print(f"\n── Blending ({LOGREG_WEIGHT:.0%} LogReg + {RF_WEIGHT:.0%} RF) ──")
    logreg_val_proba = best_logreg.predict_proba(X_val_full)[:, 1]
    blend_val_proba  = LOGREG_WEIGHT * logreg_val_proba + RF_WEIGHT * rf_val_proba
    blend_val_auc    = roc_auc_score(y_val, blend_val_proba)
    print(f"  Blend val AUC: {blend_val_auc:.4f}")

    # ── Test set evaluation ───────────────────────────────────────────────────
    logreg_test_proba = best_logreg.predict_proba(X_test_full)[:, 1]
    rf_test_proba     = rf.predict_proba(X_rf_test)[:, 1]
    blend_test_proba  = LOGREG_WEIGHT * logreg_test_proba + RF_WEIGHT * rf_test_proba

    results = {}

    print("\n" + "=" * 58)
    print("  RESULTS")
    print("=" * 58)

    # LogReg alone
    logreg_pred = (logreg_test_proba >= 0.5).astype(int)
    m = compute_metrics(y_test, logreg_pred, logreg_test_proba)
    results["logreg_finbert"] = m
    print_metrics("LogReg + FinBERT (threshold=0.5)", m)

    # Blend
    blend_pred = (blend_test_proba >= 0.5).astype(int)
    m = compute_metrics(y_test, blend_pred, blend_test_proba)
    results["blend_finbert"] = m
    print_metrics("Blend + FinBERT (threshold=0.5)", m)

    # ── Final comparison table ────────────────────────────────────────────────
    BASELINES = {
        "LogReg + TF-IDF (original)"        : 0.5251,
        "MiniLM embeddings"                 : 0.5250,
        "FinBERT-tone embeddings alone"     : 0.5324,
        "Blend + LM + filter (best so far)" : 0.5432,
    }

    print("\n" + "=" * 62)
    print("  FULL COMPARISON TABLE — Test Set AUC")
    print("=" * 62)
    for name, auc in BASELINES.items():
        marker = ""
        print(f"  {name:<45} {auc:.4f} {marker}")

    new_logreg_auc = results["logreg_finbert"]["roc_auc"]
    new_blend_auc  = results["blend_finbert"]["roc_auc"]

    best_prev = 0.5432
    marker_l  = " ✅ NEW BEST" if new_logreg_auc > best_prev else ""
    marker_b  = " ✅ NEW BEST" if new_blend_auc  > best_prev else ""

    print(f"  {'LogReg + FinBERT + market + sentiment':<45} "
          f"{new_logreg_auc:.4f}{marker_l}")
    print(f"  {'Blend + FinBERT + market + sentiment':<45} "
          f"{new_blend_auc:.4f}{marker_b}")
    print()
    print(f"  Delta (blend FinBERT vs best baseline): "
          f"{new_blend_auc - best_prev:+.4f}")
    print("=" * 62)

    # ── Save ──────────────────────────────────────────────────────────────────
    ensure_dir("artifacts/models")
    ensure_dir("artifacts/reports")

    joblib.dump(best_logreg,  "artifacts/models/finbert_blend_logreg.joblib")
    joblib.dump(rf,           "artifacts/models/finbert_blend_rf.joblib")
    joblib.dump(imputer_rf,   "artifacts/models/finbert_blend_imputer.joblib")
    joblib.dump(scaler_rf,    "artifacts/models/finbert_blend_scaler.joblib")

    save_json("artifacts/reports/metrics_finbert_blend.json", {
        "embedding_model"  : EMBEDDING_MODEL,
        "best_logreg_c"    : best_c,
        "label_threshold"  : LABEL_THRESHOLD,
        "logreg_weight"    : LOGREG_WEIGHT,
        "rf_weight"        : RF_WEIGHT,
        **{k: v for k, v in results.items()},
    })

    print("\nArtifacts saved:")
    print("  artifacts/models/finbert_blend_logreg.joblib")
    print("  artifacts/models/finbert_blend_rf.joblib")
    print("  artifacts/reports/metrics_finbert_blend.json")


if __name__ == "__main__":
    main()