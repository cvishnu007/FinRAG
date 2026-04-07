from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


POS_WORDS = {
    "beat", "beats", "benefit", "bull", "bullish", "growth", "gain",
    "gains", "positive", "strong", "upgrade", "upside", "surge",
    "record", "profit", "profits", "increase", "improve", "improves",
    "optimistic", "outperform",
}

NEG_WORDS = {
    "miss", "misses", "bear", "bearish", "decline", "drop", "drops",
    "negative", "weak", "downgrade", "downside", "plunge", "loss",
    "losses", "decrease", "concern", "concerns", "lawsuit", "probe",
    "cut", "cuts", "warning",
}


def build_text_column(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("")
    summary = df["summary"].fillna("")
    return (title + " " + summary).str.strip()


def fit_vectorizer(
    train_text: pd.Series,
    max_features: int,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_df: float,
) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
    )
    vectorizer.fit(train_text)
    return vectorizer


def transform_text(
    vectorizer: TfidfVectorizer, text: pd.Series
):
    return vectorizer.transform(text)


def fit_feature_state(train_df: pd.DataFrame) -> Dict[str, object]:
    tickers = sorted(train_df["ticker"].dropna().unique().tolist())
    dows = list(range(7))
    return {
        "tickers": tickers,
        "dows": dows,
    }


def _sentiment_features(text: pd.Series) -> pd.DataFrame:
    def score_one(doc: str) -> Tuple[float, float, float]:
        tokens = [t for t in doc.lower().split() if t.isalpha()]
        if not tokens:
            return 0.0, 0.0, 0.0
        pos = sum(1 for t in tokens if t in POS_WORDS)
        neg = sum(1 for t in tokens if t in NEG_WORDS)
        total = len(tokens)
        pos_ratio = pos / total
        neg_ratio = neg / total
        score = (pos - neg) / total
        return pos_ratio, neg_ratio, score

    scores = text.fillna("").apply(score_one)
    return pd.DataFrame(scores.tolist(), columns=["pos_ratio", "neg_ratio", "sent_score"])


def add_temporal_features(
    df: pd.DataFrame,
    timezone: str,
    market_open: str,
    market_close: str,
) -> pd.DataFrame:
    out = df.copy()
    published = pd.to_datetime(out["published"], errors="coerce")
    if published.dt.tz is None:
        published = published.dt.tz_localize("UTC")
    published = published.dt.tz_convert(timezone)

    out["dow"] = published.dt.dayofweek.astype(int)
    out["month"] = published.dt.month.astype(int)

    open_time = pd.to_datetime(market_open, format="%H:%M").time()
    close_time = pd.to_datetime(market_close, format="%H:%M").time()

    def _bucket(ts: pd.Timestamp) -> str:
        t = ts.time()
        if t < open_time:
            return "pre"
        if t > close_time:
            return "post"
        return "market"

    out["session"] = published.apply(_bucket)
    return out


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out["text"].fillna("")
    sent = _sentiment_features(text)
    out = pd.concat([out.reset_index(drop=True), sent.reset_index(drop=True)], axis=1)
    return out


def build_feature_frame(
    df: pd.DataFrame,
    timezone: str,
    market_open: str,
    market_close: str,
    use_sentiment: bool,
) -> pd.DataFrame:
    out = df.copy()
    out["text"] = build_text_column(out)
    out = add_temporal_features(out, timezone, market_open, market_close)
    if use_sentiment:
        out = add_sentiment_features(out)
    return out
