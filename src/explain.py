"""
src/explain.py
==============
Explanation generator using Groq API + FAISS retrieval.
Uses llama-3.3-70b-versatile via Groq (free tier).
"""

from __future__ import annotations

import os
from typing import List, Dict

import numpy as np
import pandas as pd

from src.retrieval import query


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    ticker: str,
    title: str,
    summary: str,
    predicted_prob: float,
    retrieved_events: List[Dict],
) -> str:
    direction  = "UP" if predicted_prob >= 0.5 else "DOWN"
    confidence = max(predicted_prob, 1 - predicted_prob) * 100

    up_count = sum(1 for e in retrieved_events if e["label"] == 1)
    avg_ret  = sum(e["return_1d"] for e in retrieved_events) / len(retrieved_events) * 100

    events_text = ""
    for i, e in enumerate(retrieved_events, 1):
        direction_str = "UP" if e["label"] == 1 else "DOWN"
        ret_pct = e["return_1d"] * 100
        events_text += (
            f"\nEvent {i}:\n"
            f"  Ticker     : {e['ticker']}\n"
            f"  Headline   : {e['title'][:150]}\n"
            f"  Date       : {e['published']}\n"
            f"  Next-day   : {ret_pct:+.2f}% ({direction_str})\n"
            f"  Similarity : {e['similarity']:.3f}\n"
        )

    prompt = f"""You are a financial analyst assistant. Your job is to explain
a stock movement prediction by citing only the provided historical analogues.
You must NOT invent facts, add general market knowledge, or speculate beyond
what the retrieved events support.

CURRENT ARTICLE
---------------
Ticker  : {ticker}
Headline: {title}
Summary : {summary[:400] if summary else "N/A"}

MODEL PREDICTION
----------------
Direction   : {direction}
Probability : {predicted_prob:.1%}
Confidence  : {confidence:.1f}%

RETRIEVED HISTORICAL ANALOGUES
-------------------------------
These are the {len(retrieved_events)} most semantically similar historical
news events from our training database, along with the actual market
reaction the following trading day:
{events_text}
Aggregate: {up_count}/{len(retrieved_events)} events were UP | Average next-day return: {avg_ret:+.2f}%

YOUR TASK
---------
Write a concise explanation (3-5 sentences) that:
1. States the prediction and confidence clearly
2. Grounds the prediction in the retrieved historical events — cite specific events by headline and date
3. Mentions the aggregate outcome of the analogues
4. Notes any important caveats (mixed signals, low similarity scores)
5. Uses ONLY the information provided above — no outside knowledge

Write in clear, professional financial analyst language."""

    return prompt


# ── Groq API call ─────────────────────────────────────────────────────────────

def call_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Call the Groq API and return the text response.
    API key is read from the GROQ_API_KEY environment variable.
    
    Free tier models available on Groq:
        llama-3.3-70b-versatile   ← best quality, use this
        llama-3.1-8b-instant      ← faster, lighter
        mixtral-8x7b-32768        ← good alternative
        gemma2-9b-it              ← Google's model
    """
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError(
            "groq package required. Install: pip install groq"
        )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set.\n"
            "Set it with:\n"
            "  Windows : set GROQ_API_KEY=gsk_your_key_here\n"
            "  Mac/Linux: export GROQ_API_KEY=gsk_your_key_here"
        )

    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst assistant. "
                    "You explain stock predictions using only the historical "
                    "evidence provided to you. You never hallucinate or add "
                    "information not present in the context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model,
        max_tokens=512,
        temperature=0.3,   # low temp = more factual, less creative
    )

    return chat_completion.choices[0].message.content


# ── Main explain function ─────────────────────────────────────────────────────

def explain_prediction(
    ticker: str,
    title: str,
    summary: str,
    article_embedding: np.ndarray,
    index,
    metadata: pd.DataFrame,
    logreg_model,
    rf_model,
    imputer,
    scaler,
    ohe,
    feature_row: pd.DataFrame,
    logreg_weight: float = 0.60,
    rf_weight: float = 0.40,
    k: int = 5,
    cross_ticker_only: bool = False,
) -> Dict:
    """
    Full pipeline: predict → retrieve → explain.
    """
    MARKET_COLS    = ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]
    SENTIMENT_COLS = ["pos_ratio", "neg_ratio", "sent_score", "surprise_score"]
    rf_cols        = MARKET_COLS + SENTIMENT_COLS

    # ── 1. Blend prediction ────────────────────────────────────────────────
    logreg_prob = float(logreg_model.predict_proba(feature_row)[:, 1][0])

    rf_features = imputer.transform(feature_row[rf_cols])
    rf_features = scaler.transform(rf_features)
    rf_prob     = float(rf_model.predict_proba(rf_features)[:, 1][0])

    blend_prob  = logreg_weight * logreg_prob + rf_weight * rf_prob

    # ── 2. Retrieve analogues ──────────────────────────────────────────────
    exclude  = ticker if cross_ticker_only else None
    retrieved = query(
        embedding    = article_embedding,
        index        = index,
        metadata     = metadata,
        k            = k,
        exclude_ticker = exclude,
    )

    # ── 3. Build prompt and call Groq ──────────────────────────────────────
    prompt = build_prompt(
        ticker           = ticker,
        title            = title,
        summary          = summary,
        predicted_prob   = blend_prob,
        retrieved_events = retrieved,
    )

    explanation = call_llm(prompt)

    return {
        "ticker"           : ticker,
        "title"            : title,
        "logreg_prob"      : round(logreg_prob, 4),
        "rf_prob"          : round(rf_prob, 4),
        "blend_prob"       : round(blend_prob, 4),
        "direction"        : "UP" if blend_prob >= 0.5 else "DOWN",
        "retrieved_events" : retrieved,
        "prompt"           : prompt,
        "explanation"      : explanation,
    }