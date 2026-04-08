import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, OneHotEncoder

from src.evaluate import compute_metrics, print_metrics
from src.features import build_feature_frame
from src.market_features import add_market_features
from src.utils import ensure_dir, load_yaml_config, save_json


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


def load_xgboost():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "XGBoost is required. Install with: pip install xgboost"
        ) from exc
    return XGBClassifier


def select_first_column(frame):
    return frame.iloc[:, 0]


def build_preprocessor(cfg: dict, use_market: bool, use_sentiment: bool):
    text_col = "text"
    cat_cols = ["ticker", "dow", "month", "session"]
    num_cols = []
    if use_sentiment:
        num_cols += ["pos_ratio", "neg_ratio", "sent_score"]
    if use_market:
        num_cols += ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]

    text_steps = [
        ("to_text", FunctionTransformer(select_first_column, validate=False)),
        (
            "tfidf",
            TfidfVectorizer(
                max_features=cfg["text"]["max_features"],
                ngram_range=tuple(cfg["text"]["ngram_range"]),
                min_df=cfg["text"]["min_df"],
                max_df=cfg["text"]["max_df"],
                stop_words="english",
            ),
        ),
    ]

    if cfg.get("text", {}).get("use_svd", False):
        text_steps.append(
            ("svd", TruncatedSVD(n_components=cfg["text"]["svd_components"], random_state=cfg["seed"]))
        )

    text_transformer = Pipeline(steps=text_steps)

    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", MaxAbsScaler()),
        ]
    )

    transformers = [("text", text_transformer, [text_col]), ("cat", cat_transformer, cat_cols)]
    if num_cols:
        transformers.append(("num", num_transformer, num_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    feature_meta = {
        "feature_columns": {
            "text": text_col,
            "categorical": cat_cols,
            "numeric": num_cols,
        },
        "market_features": use_market,
        "sentiment_features": use_sentiment,
        "text_svd": cfg.get("text", {}).get("use_svd", False),
    }
    return preprocessor, feature_meta


def main():
    config_path = os.path.join("config", "xgboost.yaml")
    cfg = load_yaml_config(config_path)

    data_path = cfg["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)

    df = pd.read_csv(data_path)
    required_cols = {"ticker", "title", "summary", "published", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published", "label"])

    label_values = set(pd.to_numeric(df["label"], errors="coerce").dropna().unique())
    if not label_values.issubset({0, 1}):
        raise ValueError(f"Unexpected label values: {sorted(label_values)}")

    df = df.sort_values("published").reset_index(drop=True)

    use_market = bool(cfg.get("features", {}).get("use_market_features", False))
    if use_market:
        market_cfg = cfg.get("market", {})
        df = add_market_features(
            df,
            prices_dir=market_cfg.get("prices_dir", "data/raw/prices"),
            timezone=market_cfg.get("timezone", "America/New_York"),
            market_close=market_cfg.get("market_close", "16:00"),
        )
    before = len(df)
    df = df.sort_values(["published", "ticker"])
    df = df.drop_duplicates(subset=["title", "published"], keep="first")
    df = df.sort_values("published").reset_index(drop=True)
    print(f"Cross-ticker dedup: {before:,} → {len(df):,} rows (removed {before - len(df):,})")

    
    train_df, val_df, test_df = time_split_per_ticker(
        df,
        train_frac=cfg["splits"]["train"],
        val_frac=cfg["splits"]["val"],
    )

    use_sentiment = bool(cfg.get("features", {}).get("use_sentiment", False))
    market_cfg = cfg.get("market", {})

    train_df = build_feature_frame(
        train_df,
        timezone=market_cfg.get("timezone", "America/New_York"),
        market_open=market_cfg.get("market_open", "09:30"),
        market_close=market_cfg.get("market_close", "16:00"),
        use_sentiment=use_sentiment,
    )
    val_df = build_feature_frame(
        val_df,
        timezone=market_cfg.get("timezone", "America/New_York"),
        market_open=market_cfg.get("market_open", "09:30"),
        market_close=market_cfg.get("market_close", "16:00"),
        use_sentiment=use_sentiment,
    )
    test_df = build_feature_frame(
        test_df,
        timezone=market_cfg.get("timezone", "America/New_York"),
        market_open=market_cfg.get("market_open", "09:30"),
        market_close=market_cfg.get("market_close", "16:00"),
        use_sentiment=use_sentiment,
    )

    preprocessor, feature_meta = build_preprocessor(cfg, use_market, use_sentiment)

    y_train = train_df["label"].astype(int)
    y_val = val_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    XGBClassifier = load_xgboost()

    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)

    best_model = None
    best_metrics = None
    best_params = None

    for params in cfg["xgboost"]["params_grid"]:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric=cfg["xgboost"]["eval_metric"],
            random_state=seed,
            n_jobs=-1,
            **params,
        )
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
        early_stop = cfg["xgboost"].get("early_stopping_rounds")
        if early_stop:
            fit_kwargs["early_stopping_rounds"] = early_stop

        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            fit_kwargs.pop("early_stopping_rounds", None)
            model.fit(X_train, y_train, **fit_kwargs)

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)

        if best_metrics is None or val_metrics["roc_auc"] > best_metrics["roc_auc"]:
            best_model = model
            best_metrics = val_metrics
            best_params = params

    model = best_model

    split_metrics = {"best_params": best_params}
    for label, X_set, y_set in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        y_pred = model.predict(X_set)
        y_proba = model.predict_proba(X_set)[:, 1]
        metrics = compute_metrics(y_set, y_pred, y_proba)
        print_metrics(label, metrics)
        split_metrics[label.lower()] = metrics

    model_path = cfg["artifacts"]["model_path"]
    preprocessor_path = cfg["artifacts"]["preprocessor_path"]
    metrics_path = cfg["artifacts"]["metrics_path"]
    compare_path = cfg["artifacts"]["compare_path"]

    ensure_dir(os.path.dirname(model_path))
    ensure_dir(os.path.dirname(preprocessor_path))

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    save_json(metrics_path, split_metrics)

    compare_payload = {"xgboost": split_metrics}
    lr_metrics_path = "artifacts/reports/metrics.json"
    if os.path.exists(lr_metrics_path):
        import json

        with open(lr_metrics_path, "r", encoding="utf-8") as f:
            compare_payload["logreg"] = json.load(f)
    save_json(compare_path, compare_payload)

    print("\nArtifacts saved:")
    print(f"  Model      : {model_path}")
    print(f"  Preprocess : {preprocessor_path}")
    print(f"  Metrics    : {metrics_path}")
    print(f"  Compare    : {compare_path}")


if __name__ == "__main__":
    main()
