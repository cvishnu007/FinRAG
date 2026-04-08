from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ── Loughran-McDonald Financial Sentiment Dictionary ──────────────────────────
# Source: Loughran & McDonald (2011) "When Is a Liability Not a Liability?"
# Journal of Finance. Purpose-built for financial/business text.
# Full dictionary: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
#
# This is a curated subset of the most impactful LM words.
# For the full 3,500-word list, download the master dictionary CSV from the
# URL above and replace these sets with pd.read_csv() lookups.

LM_POSITIVE = {
    "able", "abundance", "abundant", "acclaimed", "accomplish", "accomplished",
    "accomplishment", "accomplishments", "accurate", "achieve", "achieved",
    "achievement", "achievements", "acumen", "adaptable", "adequate",
    "admirable", "advance", "advanced", "advancement", "advantage", "advantages",
    "affirmative", "agile", "agreeable", "allay", "allaying", "ameliorate",
    "ample", "apparent", "appreciate", "appreciated", "appreciates",
    "appreciation", "appropriate", "approval", "approve", "aptitude",
    "assurance", "assure", "attain", "attained", "attractive", "authorization",
    "award", "awarded", "beat", "beats", "beneficial", "beneficially",
    "beneficiary", "benefit", "benefited", "benefits", "best", "better",
    "boost", "boosted", "breakthrough", "bull", "bullish", "capability",
    "capable", "certainty", "clarity", "comfortable", "commend",
    "commendable", "competent", "competitive", "confidence", "confident",
    "constructive", "continuity", "correct", "deliver", "delivered",
    "delivering", "dependable", "distinguished", "diversify", "dynamic",
    "earn", "earned", "earnings", "effective", "effectively", "efficiency",
    "efficient", "empower", "enhance", "enhanced", "enhances", "exceed",
    "exceeded", "exceeds", "excellence", "excellent", "exceptional",
    "exceptional", "expand", "expanded", "expanding", "expansion",
    "expedient", "experienced", "expertise", "favorable", "feasible",
    "flourish", "flourishing", "gain", "gains", "good", "great",
    "grow", "growing", "growth", "guidance", "high", "higher",
    "honest", "improve", "improved", "improvement", "improvements",
    "improves", "increasing", "increasingly", "innovative", "integrity",
    "invest", "leadership", "lucrative", "maximize", "maximizing",
    "milestone", "momentum", "new", "notable", "optimism", "optimistic",
    "outperform", "outperformed", "outperforming", "outstanding",
    "overcame", "overcome", "overperform", "positive", "positively",
    "premium", "proactive", "productive", "profitability", "profitable",
    "progress", "progressive", "profitable", "profitable", "profit",
    "profits", "promote", "promotion", "proper", "prosper", "prosperity",
    "prosperous", "raise", "raised", "rally", "rebound", "record",
    "recover", "recovering", "recovery", "reliable", "remarkable",
    "resilience", "resilient", "resolve", "reward", "rewarding",
    "rise", "rising", "robust", "safe", "secure", "stability",
    "stable", "strengthen", "strong", "stronger", "strongest", "succeed",
    "success", "successes", "successful", "successfully", "superior",
    "support", "surge", "surged", "surpass", "sustainability", "sustainable",
    "synergy", "transparent", "tremendous", "trust", "unambiuous",
    "upgrade", "upside", "valuable", "value", "viable", "win",
    "winning", "worthwhile",
}

LM_NEGATIVE = {
    "abandon", "abdicated", "aberrant", "abrupt", "absent", "abuse",
    "adversarial", "adversity", "alarm", "alleges", "allegation",
    "allegations", "ambiguity", "ambiguous", "anomaly", "anxiety",
    "arbitrary", "audit", "avoidance", "bail", "bailout", "bankrupt",
    "bankruptcy", "bear", "bearish", "bottleneck", "breach", "burden",
    "cancel", "caution", "cautious", "cease", "challenge", "challenging",
    "closure", "collapse", "complain", "complaint", "concern", "concerns",
    "confiscate", "conflict", "constrain", "contraction", "controversy",
    "corruption", "costly", "counterfeit", "crisis", "critical", "cut",
    "cuts", "cutback", "damage", "decline", "declining", "decrease",
    "deficit", "delay", "delisted", "demand", "deny", "depreciation",
    "deteriorate", "deterioration", "diminish", "disappointed",
    "disappointing", "disappointment", "disclose", "discontinue",
    "dispute", "disrupt", "disruption", "distress", "disturb", "doubt",
    "doubtful", "downgrade", "downside", "drop", "drops", "economic",
    "eliminate", "emergency", "erratic", "evade", "evasion", "excessive",
    "exhaust", "expensive", "expose", "exposure", "fail", "failed",
    "failure", "falling", "fault", "fee", "fines", "fired", "fiscal",
    "force", "foreclose", "forfeit", "fraud", "hamper", "harm", "hinder",
    "hostile", "impair", "impaired", "impairment", "impediment",
    "inability", "inadequate", "insolvent", "insufficient", "investigation",
    "irregular", "irregularity", "jeopardize", "lack", "layoff",
    "layoffs", "legal", "liability", "liabilities", "limit", "litigation",
    "loss", "losses", "low", "lower", "lowest", "misstatement",
    "manipulation", "misconduct", "miss", "misses", "missing", "misuse",
    "negative", "negligence", "nonperformance", "obstacle", "penalty",
    "plunge", "poor", "problem", "probe", "problems", "recall", "reduce",
    "reduced", "reduction", "regulatory", "reject", "replace", "resign",
    "resignation", "restatement", "restriction", "risk", "risks", "sanction",
    "scandal", "scrutiny", "setback", "shortage", "slowdown", "sluggish",
    "stagnant", "stall", "struggling", "substandard", "suffer", "suffering",
    "suspect", "suspend", "suspension", "terminate", "threat", "troubled",
    "uncertain", "uncertainty", "underperform", "underperformed",
    "unfavorable", "unstable", "violation", "volatile", "volatility",
    "warn", "warning", "weak", "weakness", "worsen", "write-down",
    "writedown", "writeoff",
}

# High-signal "earnings surprise" words — strongest predictors around
# quarterly earnings announcements
LM_SURPRISE = {
    "beat", "beats", "topped", "exceeded", "surpassed", "above",
    "outperformed", "blowout", "blew",                          # positive surprise
    "miss", "misses", "missed", "below", "fell short", "shortfall",
    "disappointed", "disappointing", "underwhelmed",             # negative surprise
    "unexpected", "surprise", "surprised", "unanticipated",
    "estimate", "estimates", "consensus", "forecast", "guidance",
    "raised guidance", "lowered guidance", "reaffirmed",
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


def transform_text(vectorizer: TfidfVectorizer, text: pd.Series):
    return vectorizer.transform(text)


def fit_feature_state(train_df: pd.DataFrame) -> Dict[str, object]:
    tickers = sorted(train_df["ticker"].dropna().unique().tolist())
    dows = list(range(7))
    return {"tickers": tickers, "dows": dows}


def _sentiment_features(text: pd.Series) -> pd.DataFrame:
    """
    Compute Loughran-McDonald sentiment scores.
    Returns pos_ratio, neg_ratio, sent_score, and surprise_score.
    """
    def score_one(doc: str) -> Tuple[float, float, float, float]:
        tokens = [t for t in doc.lower().split() if t.isalpha()]
        if not tokens:
            return 0.0, 0.0, 0.0, 0.0

        total = len(tokens)
        pos  = sum(1 for t in tokens if t in LM_POSITIVE)
        neg  = sum(1 for t in tokens if t in LM_NEGATIVE)
        surp = sum(1 for t in tokens if t in LM_SURPRISE)

        pos_ratio    = pos  / total
        neg_ratio    = neg  / total
        sent_score   = (pos - neg) / total
        surp_score   = surp / total

        return pos_ratio, neg_ratio, sent_score, surp_score

    scores = text.fillna("").apply(score_one)
    return pd.DataFrame(
        scores.tolist(),
        columns=["pos_ratio", "neg_ratio", "sent_score", "surprise_score"],
    )


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

    out["dow"]   = published.dt.dayofweek.astype(int)
    out["month"] = published.dt.month.astype(int)

    # Earnings season flag: Jan/Apr/Jul/Oct are peak reporting months
    out["earnings_season"] = published.dt.month.isin([1, 4, 7, 10]).astype(int)

    open_time  = pd.to_datetime(market_open,  format="%H:%M").time()
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
    out = pd.concat(
        [out.reset_index(drop=True), sent.reset_index(drop=True)], axis=1
    )
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
