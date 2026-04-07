import os

import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, OneHotEncoder

from src.evaluate import compute_metrics, print_metrics
from src.features import build_feature_frame
from src.market_features import add_market_features
from src.utils import ensure_dir, load_yaml_config, save_json


def _extract_first_col(x):
    """Extract the first column from a DataFrame — used by FunctionTransformer."""
    return x.iloc[:, 0]


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    n_total = len(df)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Split resulted in empty set. "
            "Adjust split fractions or check dataset size."
        )

    return train_df, val_df, test_df


def main():
    config_path = os.path.join("config", "baseline.yaml")
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

    train_df, val_df, test_df = time_split(
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

    text_col = "text"
    cat_cols = ["ticker", "dow", "month", "session"]
    num_cols = []
    if use_sentiment:
        num_cols += ["pos_ratio", "neg_ratio", "sent_score"]
    if use_market:
        num_cols += ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]

    text_transformer = Pipeline(
        steps=[
            ("to_text", FunctionTransformer(_extract_first_col, validate=False)),
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
    )

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

    y_train = train_df["label"].astype(int)
    y_val = val_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    penalties = cfg["model"].get("penalties", ["l2"])
    c_values = cfg["model"].get("c_values", [1.0])
    solver = cfg["model"].get("solver", "lbfgs")

    best_model = None
    best_metrics = None
    best_key = None

    for penalty in penalties:
        for c_val in c_values:
            clf = LogisticRegression(
                max_iter=cfg["model"]["max_iter"],
                class_weight=cfg["model"]["class_weight"],
                random_state=seed,
                n_jobs=None,
                penalty=penalty,
                C=c_val,
                solver=solver,
            )
            model = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
            model.fit(train_df, y_train)

            y_val_pred = model.predict(val_df)
            y_val_proba = model.predict_proba(val_df)[:, 1]
            val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)

            if best_metrics is None or val_metrics["roc_auc"] > best_metrics["roc_auc"]:
                best_model = model
                best_metrics = val_metrics
                best_key = {"penalty": penalty, "C": c_val}

    model = best_model

    split_metrics = {"best_params": best_key}
    for label, x_set, y_set in [
        ("Train", train_df, y_train),
        ("Validation", val_df, y_val),
        ("Test", test_df, y_test),
    ]:
        y_pred = model.predict(x_set)
        y_proba = model.predict_proba(x_set)[:, 1]
        metrics = compute_metrics(y_set, y_pred, y_proba)
        print_metrics(label, metrics)
        split_metrics[label.lower()] = metrics

    model_path = cfg["artifacts"]["model_path"]
    vectorizer_path = cfg["artifacts"]["vectorizer_path"]
    feature_meta_path = cfg["artifacts"]["feature_meta_path"]
    metrics_path = cfg["artifacts"]["metrics_path"]

    ensure_dir(os.path.dirname(model_path))
    ensure_dir(os.path.dirname(vectorizer_path))
    ensure_dir(os.path.dirname(feature_meta_path))

    joblib.dump(model, model_path)

    tfidf = model.named_steps["preprocess"].named_transformers_["text"].named_steps["tfidf"]
    joblib.dump(tfidf, vectorizer_path)

    feature_meta = {
        "feature_columns": {
            "text": text_col,
            "categorical": cat_cols,
            "numeric": num_cols,
        },
        "market_features": use_market,
        "sentiment_features": use_sentiment,
        "market_timezone": market_cfg.get("timezone", "America/New_York"),
    }
    joblib.dump(feature_meta, feature_meta_path)

    save_json(metrics_path, split_metrics)

    print("\nArtifacts saved:")
    print(f"  Model      : {model_path}")
    print(f"  Vectorizer : {vectorizer_path}")
    print(f"  Features   : {feature_meta_path}")
    print(f"  Metrics    : {metrics_path}")


if __name__ == "__main__":
    main()
