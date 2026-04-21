"""
src/explain.py
==============
Explanation generator using Groq API + FAISS retrieval.
Uses llama-3.3-70b-versatile via Groq (free tier).

Changes from original:
  - build_prompt        : pre-fills ALL numbers into the JSON template
                          so the LLM only writes free-text fields
  - _clean_and_parse_json: 6-strategy robust JSON cleaner that handles
                          markdown fences, trailing commas, None/True/False,
                          inline comments, and preamble text
  - verify_explanation  : checks every verifiable factual claim against
                          ground-truth metadata (7 checks)
  - generate_template_explanation: 100% factually-safe fallback with no LLM
  - explain_prediction  : retry loop with error feedback + template fallback
  - call_llm            : max_tokens raised to 1500
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Dict

import numpy as np
import pandas as pd

from src.retrieval import query


# ── JSON cleaner ──────────────────────────────────────────────────────────────

def _clean_and_parse_json(raw: str) -> dict:
    """
    Attempts multiple strategies to extract valid JSON from an LLM response
    that may contain markdown fences, trailing commas, preamble text, or
    Python-style literals.

    Strategies applied in order:
      1. Strip markdown code fences  (``` or ```json)
      2. Extract outermost { } block (handles preamble / postamble text)
      3. Remove trailing commas before } or ]
      4. Replace Python None  → null
      5. Replace Python True/False → true/false
      6. Remove // inline comments
    """
    text = raw.strip()

    # ── Strategy 1: strip markdown code fences ────────────────────────────
    if "```" in text:
        parts = text.split("```")
        # content is between the first and second fence
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    # ── Strategy 2: find outermost { } block ──────────────────────────────
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start : brace_end + 1]

    # ── Strategy 3: remove trailing commas ────────────────────────────────
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # ── Strategy 4: Python None → JSON null ───────────────────────────────
    text = re.sub(r":\s*None\b", ": null", text)

    # ── Strategy 5: Python True/False → JSON true/false ───────────────────
    text = re.sub(r":\s*True\b",  ": true",  text)
    text = re.sub(r":\s*False\b", ": false", text)

    # ── Strategy 6: remove // inline comments ─────────────────────────────
    text = re.sub(r"//[^\n]*", "", text)

    return json.loads(text)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    ticker: str,
    title: str,
    summary: str,
    predicted_prob: float,
    retrieved_events: List[Dict],
) -> str:
    """
    Builds a structured prompt that pre-fills every verifiable number
    directly into the JSON template. The LLM only needs to write three
    types of free-text fields:
      - relevance   (one sentence per event)
      - caveats     (one sentence)
      - summary     (2-3 sentences)

    Pre-filling numbers eliminates the most common hallucination modes:
    wrong return sign, wrong direction, wrong aggregate stats.
    """
    direction  = "UP" if predicted_prob >= 0.5 else "DOWN"
    confidence = max(predicted_prob, 1 - predicted_prob) * 100

    up_count = sum(1 for e in retrieved_events if e["label"] == 1)
    avg_ret  = sum(e["return_1d"] for e in retrieved_events) / len(retrieved_events) * 100

    # ── Build human-readable events block (for the context section) ───────
    events_text = ""
    for i, e in enumerate(retrieved_events, 1):
        direction_str = "UP" if e["label"] == 1 else "DOWN"
        ret_pct = e["return_1d"] * 100
        events_text += (
            f"\nEvent {i}:\n"
            f"  Ticker     : {e['ticker']}\n"
            f"  Headline   : {e['title'][:150]}\n"
            f"  Date       : {str(e['published'])[:10]}\n"
            f"  Next-day   : {ret_pct:+.2f}% ({direction_str})\n"
            f"  Similarity : {e['similarity']:.3f}\n"
        )

    # ── Build pre-filled JSON template ────────────────────────────────────
    # Every number is computed by Python — the LLM copies them verbatim.
    # Only FILL_IN fields require the LLM to generate language.
    event_entries = []
    for i, e in enumerate(retrieved_events, 1):
        ret_pct       = round(e["return_1d"] * 100, 2)
        dir_str       = "UP" if e["label"] == 1 else "DOWN"
        date_str      = str(e["published"])[:10]
        ticker_str    = e["ticker"]
        entry = (
            f"    {{\n"
            f'      "rank": {i},\n'
            f'      "ticker": "{ticker_str}",\n'
            f'      "date": "{date_str}",\n'
            f'      "direction_cited": "{dir_str}",\n'
            f'      "return_cited_pct": {ret_pct},\n'
            f'      "relevance": "FILL_IN one sentence why this is analogous"\n'
            f"    }}"
        )
        event_entries.append(entry)

    events_json_block = ",\n".join(event_entries)

    prompt = f"""You are a financial analyst assistant.
Explain the stock prediction below using ONLY the retrieved historical events.
Do NOT change any numbers. Do NOT add outside knowledge.

CURRENT ARTICLE
---------------
Ticker  : {ticker}
Headline: {title}
Summary : {summary[:400] if summary else "N/A"}

MODEL PREDICTION
----------------
Direction   : {direction}
Probability : {predicted_prob:.4f}
Confidence  : {confidence:.1f}%

RETRIEVED HISTORICAL ANALOGUES
-------------------------------
{events_text}
Aggregate: {up_count}/{len(retrieved_events)} events were UP | Average return: {avg_ret:+.2f}%

YOUR TASK
---------
Return ONLY the JSON object below.
- Replace every "FILL_IN ..." string with your own language.
- Do NOT change any number, ticker, date, or direction value.
- No text before the opening brace.
- No text after the closing brace.
- No trailing commas.
- No markdown, no backticks.

{{
  "direction": "{direction}",
  "confidence_pct": {round(confidence, 1)},
  "events_cited": [
{events_json_block}
  ],
  "aggregate_up_count": {up_count},
  "aggregate_avg_return_pct": {round(avg_ret, 2)},
  "caveats": "FILL_IN one sentence about uncertainty or mixed signals",
  "summary": "FILL_IN 2-3 sentences explaining the prediction citing the events above"
}}"""

    return prompt


# ── Explanation verifier ──────────────────────────────────────────────────────

def verify_explanation(
    parsed: dict,
    retrieved_events: List[Dict],
    blend_prob: float,
) -> dict:
    """
    Checks every verifiable factual claim in the parsed JSON against
    ground-truth metadata from the retrieved events.

    Hard errors  → valid=False  (factual mistakes that matter)
    Warnings     → valid=True   (minor discrepancies, logged but not fatal)

    Checks:
      1. Overall direction matches blend_prob
      2. Confidence % matches blend_prob within 5% tolerance
      3. Every cited event actually exists in the retrieved top-5
      4. Each event's direction_cited matches actual label
      5. Each event's return_cited_pct matches actual return within 0.5%
      6. aggregate_up_count matches actual count  (warning)
      7. aggregate_avg_return_pct matches actual avg within 0.5%  (warning)
    """
    errors   = []
    warnings = []

    # ── Check 1: direction matches blend_prob ─────────────────────────────
    expected_direction = "UP" if blend_prob >= 0.5 else "DOWN"
    if parsed.get("direction") != expected_direction:
        errors.append(
            f"Direction mismatch: explanation says '{parsed.get('direction')}' "
            f"but model predicted '{expected_direction}' "
            f"(blend_prob={blend_prob:.4f})"
        )

    # ── Check 2: confidence_pct matches blend_prob within 5% ─────────────
    expected_conf = max(blend_prob, 1 - blend_prob) * 100
    stated_conf   = parsed.get("confidence_pct", -1)
    if abs(stated_conf - expected_conf) > 5.0:
        errors.append(
            f"Confidence mismatch: explanation says {stated_conf:.1f}% "
            f"but blend_prob implies {expected_conf:.1f}%"
        )

    # ── Build ground-truth lookup keyed by (TICKER, YYYY-MM-DD) ──────────
    event_lookup: Dict[tuple, dict] = {}
    for e in retrieved_events:
        date_key = str(e["published"])[:10]
        key = (e["ticker"].upper(), date_key)
        event_lookup[key] = {
            "return_pct": round(e["return_1d"] * 100, 4),
            "direction" : "UP" if e["label"] == 1 else "DOWN",
            "label"     : e["label"],
        }

    # ── Checks 3-5: per-event validation ─────────────────────────────────
    for cited in parsed.get("events_cited", []):
        ticker_cited = str(cited.get("ticker", "")).upper()
        date_cited   = str(cited.get("date",   ""))[:10]
        key          = (ticker_cited, date_cited)

        # Check 3: event must exist in retrieved top-5
        if key not in event_lookup:
            errors.append(
                f"Hallucinated event: ({ticker_cited}, {date_cited}) "
                f"was NOT in the retrieved top-5"
            )
            continue

        ground = event_lookup[key]

        # Check 4: direction matches actual label
        if cited.get("direction_cited") != ground["direction"]:
            errors.append(
                f"Direction error for {ticker_cited} on {date_cited}: "
                f"explanation says '{cited.get('direction_cited')}' "
                f"but actual outcome was '{ground['direction']}'"
            )

        # Check 5: return percentage within 0.5% tolerance
        stated_return = cited.get("return_cited_pct")
        if stated_return is not None:
            if abs(float(stated_return) - ground["return_pct"]) > 0.5:
                errors.append(
                    f"Return error for {ticker_cited} on {date_cited}: "
                    f"explanation cites {float(stated_return):+.2f}% "
                    f"but actual was {ground['return_pct']:+.2f}%"
                )

    # ── Check 6: aggregate_up_count (warning only) ────────────────────────
    actual_up  = sum(1 for e in retrieved_events if e["label"] == 1)
    stated_up  = parsed.get("aggregate_up_count", -1)
    if stated_up != actual_up:
        warnings.append(
            f"Aggregate UP count: explanation says {stated_up} "
            f"but actual is {actual_up}"
        )

    # ── Check 7: aggregate_avg_return_pct within 0.5% (warning only) ─────
    actual_avg = (
        sum(e["return_1d"] for e in retrieved_events)
        / len(retrieved_events) * 100
    )
    stated_avg = parsed.get("aggregate_avg_return_pct")
    if stated_avg is not None:
        if abs(float(stated_avg) - actual_avg) > 0.5:
            warnings.append(
                f"Aggregate return: explanation says {float(stated_avg):+.2f}% "
                f"but actual is {actual_avg:+.2f}%"
            )

    return {
        "valid"   : len(errors) == 0,
        "errors"  : errors,
        "warnings": warnings,
    }


# ── Template fallback ─────────────────────────────────────────────────────────

def generate_template_explanation(
    ticker: str,
    blend_prob: float,
    retrieved_events: List[Dict],
) -> str:
    """
    Generates a factually guaranteed explanation directly from retrieved
    event metadata. No LLM involved — zero hallucination risk.

    Called when the LLM fails verification after all retries.
    Every number in the output comes directly from the metadata dict,
    never from the LLM.
    """
    direction  = "upward"   if blend_prob >= 0.5 else "downward"
    confidence = max(blend_prob, 1 - blend_prob) * 100
    up_count   = sum(1 for e in retrieved_events if e["label"] == 1)
    avg_return = (
        sum(e["return_1d"] for e in retrieved_events)
        / len(retrieved_events) * 100
    )

    top         = retrieved_events[0]          # highest cosine similarity
    top_dir     = "gained" if top["label"] == 1 else "fell"
    top_ret_abs = abs(top["return_1d"] * 100)
    top_date    = str(top["published"])[:10]

    event_lines = []
    for e in retrieved_events:
        d   = "UP" if e["label"] == 1 else "DOWN"
        ret = e["return_1d"] * 100
        event_lines.append(
            f"{e['ticker']} ({str(e['published'])[:10]}): {ret:+.2f}% {d}"
        )
    events_summary = " | ".join(event_lines)

    return (
        f"The model predicts {direction} movement for {ticker} with "
        f"{confidence:.1f}% confidence. "
        f"The most semantically similar historical event was "
        f"'{top['title'][:80]}' ({top['ticker']}, {top_date}), "
        f"after which the stock {top_dir} {top_ret_abs:.2f}% the next day. "
        f"Across the 5 most similar historical events, {up_count} of 5 "
        f"resulted in gains with an average next-day return of {avg_return:+.2f}%. "
        f"Historical outcomes: {events_summary}."
    )


# ── Groq API call ─────────────────────────────────────────────────────────────

def call_llm(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """
    Call the Groq API and return the raw text response.
    API key is read from the GROQ_API_KEY environment variable.

    max_tokens raised to 1500 to prevent JSON truncation when
    the response contains 5 events with full relevance sentences.
    """
    try:
        from groq import Groq
    except ImportError:
        raise RuntimeError("groq package required. Install: pip install groq")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set.\n"
            "  Windows : $env:GROQ_API_KEY='gsk_...'\n"
            "  Mac/Linux: export GROQ_API_KEY=gsk_..."
        )

    client = Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analyst assistant. "
                    "You explain stock predictions using only the historical "
                    "evidence provided. You never hallucinate, never change "
                    "numbers, and never add information not present in the context. "
                    "You always respond with valid JSON and nothing else."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=model,
        max_tokens=1500,   # raised from 512 — prevents mid-JSON truncation
        temperature=0.3,
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
    logreg_weight:     float = 0.60,
    rf_weight:         float = 0.40,
    k:                 int   = 5,
    cross_ticker_only: bool  = False,
    max_retries:       int   = 3,
    debug:             bool  = False,   # set True to print raw LLM responses
) -> Dict:
    """
    Full pipeline: predict → retrieve → explain with verification.

    Flow:
      A. Compute blend probability (LogReg + RF)
      B. Retrieve top-k analogues from FAISS
      C. Build pre-filled JSON prompt
      D. Call LLM with retry loop:
           - parse JSON using robust cleaner
           - verify all factual claims against metadata
           - on failure: append error feedback and retry
      E. If all retries fail: use template fallback (no LLM, always correct)
      F. Return full result dict with verification metadata

    Returns
    -------
    dict with keys:
      ticker, title, logreg_prob, rf_prob, blend_prob, direction,
      retrieved_events, prompt, explanation,
      verified, used_template, attempts, verification_errors, parsed_json
    """
    MARKET_COLS    = ["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]
    SENTIMENT_COLS = ["pos_ratio", "neg_ratio", "sent_score", "surprise_score"]
    rf_cols        = MARKET_COLS + SENTIMENT_COLS

    # ── A: Blend prediction ───────────────────────────────────────────────
    logreg_prob = float(logreg_model.predict_proba(feature_row)[:, 1][0])

    rf_features = imputer.transform(feature_row[rf_cols])
    rf_features = scaler.transform(rf_features)
    rf_prob     = float(rf_model.predict_proba(rf_features)[:, 1][0])

    blend_prob  = logreg_weight * logreg_prob + rf_weight * rf_prob

    # ── B: Retrieve analogues ─────────────────────────────────────────────
    exclude   = ticker if cross_ticker_only else None
    retrieved = query(
        embedding      = article_embedding,
        index          = index,
        metadata       = metadata,
        k              = k,
        exclude_ticker = exclude,
    )

    # ── C: Build prompt ───────────────────────────────────────────────────
    prompt = build_prompt(
        ticker           = ticker,
        title            = title,
        summary          = summary,
        predicted_prob   = blend_prob,
        retrieved_events = retrieved,
    )

    # ── D: LLM call with retry loop ───────────────────────────────────────
    last_errors     = []
    verified_result = None
    used_template   = False
    attempts        = 0

    for attempt in range(max_retries):
        attempts     = attempt + 1
        raw_response = call_llm(prompt)

        if debug:
            print(f"\n{'='*60}")
            print(f"ATTEMPT {attempts} RAW LLM RESPONSE:")
            print(raw_response)
            print(f"{'='*60}\n")

        # ── Parse JSON using robust cleaner ───────────────────────────
        try:
            parsed = _clean_and_parse_json(raw_response)
        except json.JSONDecodeError as e:
            last_errors = [f"JSON parse failed: {e}"]

            # Show exactly where parsing failed to help debug
            fail_pos = getattr(e, "pos", 0)
            snippet  = raw_response[max(0, fail_pos - 50) : fail_pos + 50]
            print(f"  [attempt {attempts}] JSON parse error at pos {fail_pos}:")
            print(f"  ...{repr(snippet)}...")

            prompt += (
                f"\n\nATTEMPT {attempts} REJECTED — JSON PARSE ERROR: {e}\n"
                f"STRICT RULES FOR NEXT ATTEMPT:\n"
                f"1. Start your response with {{ and end with }}\n"
                f"2. No text, no explanation outside the JSON\n"
                f"3. No trailing commas after the last item in any list or object\n"
                f"4. Use null not None, use true/false not True/False\n"
                f"5. All string values must be in double quotes\n"
                f"Return ONLY the corrected JSON object."
            )
            continue

        # ── Verify all factual claims ──────────────────────────────────
        verification = verify_explanation(parsed, retrieved, blend_prob)

        if verification["valid"]:
            verified_result = parsed
            last_errors     = []

            if verification["warnings"]:
                print(f"  [attempt {attempts}] Verified with warnings:")
                for w in verification["warnings"]:
                    print(f"    ⚠ {w}")
            break

        else:
            last_errors = verification["errors"]
            error_list  = "\n".join(f"  - {err}" for err in last_errors)
            print(f"  [attempt {attempts}] Verification failed:")
            for err in last_errors:
                print(f"    ✗ {err}")

            prompt += (
                f"\n\nATTEMPT {attempts} REJECTED — FACTUAL ERRORS FOUND:\n"
                f"{error_list}\n\n"
                f"Fix ALL errors above. Rules:\n"
                f"- Copy return numbers EXACTLY from the RETRIEVED HISTORICAL "
                f"ANALOGUES section — do not round differently or flip signs\n"
                f"- Copy directions EXACTLY as shown (UP or DOWN)\n"
                f"- Copy dates EXACTLY in YYYY-MM-DD format\n"
                f"Return ONLY the corrected JSON."
            )

    # ── E: Fallback if all retries failed ─────────────────────────────────
    if verified_result is None:
        explanation   = generate_template_explanation(ticker, blend_prob, retrieved)
        used_template = True
        print(
            f"  ⚠ LLM failed verification after {max_retries} attempts "
            f"for {ticker}. Using template fallback.\n"
            f"  Final errors: {last_errors}"
        )
    else:
        explanation = verified_result.get("summary", "")

    # ── F: Return full result ─────────────────────────────────────────────
    return {
        "ticker"             : ticker,
        "title"              : title,
        "logreg_prob"        : round(logreg_prob,  4),
        "rf_prob"            : round(rf_prob,       4),
        "blend_prob"         : round(blend_prob,    4),
        "direction"          : "UP" if blend_prob >= 0.5 else "DOWN",
        "retrieved_events"   : retrieved,
        "prompt"             : prompt,
        "explanation"        : explanation,
        "verified"           : not used_template,
        "used_template"      : used_template,
        "attempts"           : attempts,
        "verification_errors": last_errors,
        "parsed_json"        : verified_result,
    }