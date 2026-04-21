# test_verification.py — run this first to check everything works

import pandas as pd
import joblib
import numpy as np
from src.embeddings import encode_texts
from src.retrieval import load_index
from src.explain import explain_prediction
from src.features import build_feature_frame, build_text_column
from src.market_features import add_market_features
from scripts.train_blend import _extract_first_col

DATA_PATH  = "data/processed/master_dataset.csv"
PRICES_DIR = "data/raw/prices"
CACHE_PATH = "artifacts/cache/embeddings.pkl"

df = pd.read_csv(DATA_PATH)
df["published"] = pd.to_datetime(df["published"])
df = df.dropna(subset=["published", "label"])
df = add_market_features(df, PRICES_DIR, "America/New_York", "16:00")
df = build_feature_frame(df, "America/New_York", "09:30", "16:00", use_sentiment=True)
df["text"] = build_text_column(df)

sample = df.sample(1, random_state=42).reset_index(drop=True)

logreg = joblib.load("artifacts/models/logreg_blend.joblib")
rf     = joblib.load("artifacts/models/rf_blend.joblib")
imp    = joblib.load("artifacts/models/rf_imputer.joblib")
scl    = joblib.load("artifacts/models/rf_scaler.joblib")
index, metadata = load_index()

emb = encode_texts(sample["text"], cache_path=CACHE_PATH, show_progress=False)

result = explain_prediction(
    ticker            = sample.iloc[0]["ticker"],
    title             = sample.iloc[0]["title"],
    summary           = sample.iloc[0].get("summary", ""),
    article_embedding = emb[0],
    index             = index,
    metadata          = metadata,
    logreg_model      = logreg,
    rf_model          = rf,
    imputer           = imp,
    scaler            = scl,
    ohe               = None,
    feature_row       = sample,
    max_retries       = 3,
)

print(f"Verified     : {result['verified']}")
print(f"Used template: {result['used_template']}")
print(f"Attempts     : {result['attempts']}")
print(f"Errors       : {result['verification_errors']}")
print(f"\nExplanation:\n{result['explanation']}")