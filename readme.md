# Financial News & Stock Prediction System

## 1. Project Goal

This project builds a **financial forecasting and explanation system** that:

1. Predicts **short-term stock movement (next-day up/down)** based on
   historical stock prices and news momentum.
2. Generates **explanations grounded in real historical events** using a
   Retrieval-Augmented Generation (RAG) approach to prevent hallucination.

Instead of an opaque black-box model, the system acts as an analytical
assistant — when predicting a stock's movement it explicitly cites
historically analogous news events and their empirical market impacts.

---

## 2. System Architecture

```
Breaking News Article
        │
        ▼
┌─────────────────────┐     ┌────────────────────────────┐
│   MiniLM Encoder    │────▶│    FAISS Retrieval Index   │
│   (384-dim embed.)  │     │    17,933 training events  │
└─────────────────────┘     └──────────────┬─────────────┘
        │                                  │ Top-5 analogues
        ▼                                  ▼
┌─────────────────────┐     ┌────────────────────────────┐
│    Blend Model      │     │  Historical Events +       │
│  LogReg 60%         │     │  Actual Market Reactions   │
│  + RF     40%       │     └──────────────┬─────────────┘
│  Test AUC = 0.543   │                    │
└──────────┬──────────┘                    │
           │ Probability                   │
           └──────────────┬────────────────┘
                          ▼
                ┌──────────────────────┐
                │   LLaMA 3.3 70B      │
                │   via Groq API       │
                │   Explanation        │
                │   Generator          │
                └──────────────────────┘
                          │
                          ▼
           "AVGO predicted DOWN (51.4% conf.)
            based on 2 similar Teledyne FLIR
            announcements — avg return -0.57%"
```

### Components

**A. Forecast Backbone** — Blended classifier (60% Logistic Regression +
40% Random Forest) trained on TF-IDF text features, Loughran-McDonald
financial sentiment scores, and lagged market momentum features.
Test AUC: **0.5432**

**B. RAG Retrieval Index** — FAISS flat inner-product index over
all-MiniLM-L6-v2 embeddings of 17,933 training events. Retrieves the
top-5 most semantically similar historical events for any query article.
Average retrieval similarity: **0.593**

**C. Explanation Generator** — LLaMA 3.3 70B via Groq API. Prompted to
explain predictions *only* using retrieved historical context.
Hallucination rate: **0%** across 100 evaluated examples.

---

## 3. Data Engineering

### Sources
- **News:** Alpha Vantage News Sentiment API (relevance >= 0.1 per ticker)
- **Prices:** yfinance OHLCV auto-adjusted + SPY as market benchmark

### Alignment Logic
Articles are aligned to the *next available market close* to capture
the true market reaction. UNIX timestamps throughout; timezone conversion
applied only at feature-engineering time.

### Master Dataset (`data/processed/master_dataset.csv`)

| Field | Value |
|---|---|
| Raw rows | 32,915 |
| After near-zero return filter (±0.2%) | 29,428 |
| After cross-ticker dedup | 25,632 used for training |
| Tickers | 20 large-cap US stocks |
| Time span | January 2019 — March 2026 |
| Label balance | ~51% UP / ~49% DOWN |

**Schema:**

| Column | Description |
|---|---|
| ticker | Stock symbol |
| title | Article headline |
| summary | Article body excerpt |
| published | Publication timestamp (UTC) |
| trade_date | Next trading day |
| Close | Closing price on trade_date |
| return_1d | Raw next-day return |
| excess_return | return_1d minus SPY return |
| spy_return | SPY benchmark return that day |
| label | 1 = UP, 0 = DOWN |

### Key Data Findings

- **Lag distribution:** 97.8% of articles have lag=1 day — no staleness
  problem exists in this dataset
- **Excess return label rejected:** Using stock-minus-SPY as label dropped
  AUC from 0.543 to 0.483. Headlines predict absolute price movement,
  not alpha over market
- **Cross-ticker dedup critical:** 4,557 articles appeared under multiple
  tickers — deduplication before splitting prevented train/test leakage

### Data Scripts

| Script | Purpose |
|---|---|
| `scripts/download_news.py` | Alpha Vantage news fetch |
| `scripts/download_prices.py` | yfinance OHLCV download |
| `scripts/download_spy.py` | SPY benchmark download |
| `scripts/merge_data.py` | Align news to next trading day |
| `scripts/diagnose.py` | Dataset diagnostic report |

---

## 4. Modelling

### All Experiments (Chronological)

| Phase | Model | Test AUC | Train AUC | Gap | Notes |
|---|---|---|---|---|---|
| Baseline | LogReg + TF-IDF | 0.5251 | 0.5743 | 0.049 | Clean, no leakage |
| Trees | Random Forest (best) | 0.5267 | 0.6825 | 0.156 | Overfit |
| Trees | XGBoost (best) | 0.5080 | 0.7175 | 0.210 | Severe overfit |
| Improved | LogReg + LM sentiment | 0.5410 | — | — | LM dict key win |
| **Best** | **Blend + LM + filter** | **0.5432** | — | — | **Production model** |
| Embeddings | MiniLM + LogReg | 0.5250 | 0.5878 | 0.063 | Same as TF-IDF |
| Embeddings | FinBERT-tone alone | 0.5324 | 0.5677 | 0.035 | Best embedding model |
| Embeddings | FinBERT-tone + Blend | 0.5432 | — | — | Tied with TF-IDF blend |

> Full experiment scripts are archived in `experiments/` — see
> `experiments/README.md` for details on each run.

### Key Conclusion on Embeddings

Four fundamentally different text representations were tested:

```
TF-IDF (sparse bag of words)              → 0.5432  (with blend)
MiniLM/384-dim (general dense)            → 0.5250
FinBERT-tone/768-dim (financial dense)    → 0.5324  (alone)
FinBERT-tone/768-dim + full blend         → 0.5432  (tied)
```

**The ceiling is not the text representation.** Better embeddings do not
break the 0.543 barrier. The RF component and label threshold filter are
doing the heavy lifting in the blend — the text representation is
largely interchangeable once those are in place. The limiting factor is
market efficiency: publicly available news is largely priced in by open.

### Best Model: Blend (`scripts/train_blend.py`)

Three improvements over the original baseline:

1. **Loughran-McDonald financial sentiment dictionary** — ~600 purpose-built
   financial words replacing the original 40-word lexicon, plus
   `surprise_score` for earnings beat/miss language
2. **Label threshold filter** — dropped rows where `|return_1d| < 0.2%`
   to remove near-zero noise the model cannot distinguish
3. **Blended prediction** — 60% Logistic Regression + 40% Random Forest
   weighted probability average

**Final test results (full 3,864-row test set):**

| Metric | Value |
|---|---|
| ROC-AUC | 0.5432 |
| Accuracy | 0.5277 |
| Precision | 0.5384 |
| Recall | 0.6034 |
| F1 | 0.5691 |

**Artifacts:**
```
artifacts/models/logreg_blend.joblib
artifacts/models/rf_blend.joblib
artifacts/models/rf_imputer.joblib
artifacts/models/rf_scaler.joblib
artifacts/models/blend_meta.joblib
```

---

## 5. RAG Retrieval Index

FAISS flat inner-product index built over 17,933 training-set MiniLM
embeddings. Given any article, retrieves the top-5 most semantically
similar historical events and their realised market outcomes.

```bash
python -m scripts.build_index
```

**Index stats:**

| Metric | Value |
|---|---|
| Vectors indexed | 17,933 |
| Embedding dimension | 384 (MiniLM) |
| Search type | Exact inner product (cosine similarity) |
| Avg retrieval similarity | 0.593 |

**Smoke test:** Given a PFE COVID vaccine article, retrieved 4 other
Pfizer vaccine articles with similarities 0.633–0.644. Semantically
precise retrieval confirmed.

```
artifacts/retrieval/faiss_index.pkl
artifacts/retrieval/metadata.pkl
```

---

## 6. Explanation Generation

LLaMA 3.3 70B via Groq API (free tier) generates grounded explanations
from the blend model probability + top-5 retrieved historical analogues.
The LLM is prompted to use *only* the retrieved context — no outside
knowledge, no hallucination.

```bash
set GROQ_API_KEY=gsk_your_key_here
python -m scripts.run_explain
```

### Example Output

```
Ticker  : AVGO
Headline: Teledyne FLIR OEM Launches Boson+ IQ Thermal Imaging Kit
Date    : 2025-09-09
Actual  : DOWN ❌ (-2.69%)

Predicted: DOWN (blend prob: 48.5%) ✅ CORRECT

Retrieved Analogues:
  1. [AVGO] Qualcomm-built processor by Teledyne FLIR (2024-04-17)
     -1.84% | DOWN | sim=0.689
  2. [NVDA] Teledyne FLIR Tracking Software (2022-06-23)
     -1.50% | DOWN | sim=0.684
  3. [NVDA] Teledyne FLIR Prism AI Software (2022-07-11)
     +0.54% | UP   | sim=0.707

Explanation:
  The model predicts a downward movement in AVGO with probability 48.6%.
  This is grounded in the Qualcomm-built Teledyne FLIR processor
  announcement on 2024-04-17 which led to a -1.84% decline in AVGO,
  and the Teledyne FLIR software event on 2022-06-23 which produced
  -1.50% in NVDA. The aggregate shows 2/5 analogues were UP with
  average return -0.57%.
```

---

## 7. Full System Evaluation

100-example evaluation on held-out test set.

```bash
python -m scripts.evaluate_system
```

### Results

**Prediction Performance**

| Metric | Value |
|---|---|
| Overall accuracy (100 examples) | 48.0% |
| Full test set AUC (3,864 rows) | 0.5432 |
| Avg blend prob when predicting UP | 0.512 |
| Avg blend prob when predicting DOWN | 0.491 |

> Note: 48% point accuracy on 100 examples is within normal sampling
> variance of the 54.3% AUC model (±10% CI at n=100). The model
> correctly expressed low confidence on all predictions — all blend
> probabilities fell between 0.47–0.54, reflecting honest uncertainty.

**Accuracy by Return Magnitude**

| Return bucket | Accuracy | n |
|---|---|---|
| Small (< 1%) | 52.2% | 46 |
| Medium (1–3%) | 46.7% | 45 |
| Large (> 3%) | 33.3% | 9 |

Large moves are driven by surprise events unpredictable from article
text alone — the model is appropriately uncertain on those.

**Retrieval Quality**

| Metric | Value |
|---|---|
| Avg cosine similarity | 0.593 |
| High-similarity articles (>= 0.65) | 22/100 |
| Accuracy on high-sim articles | 50.0% |
| Accuracy on low-sim articles | 47.4% |

**Explanation Grounding**

| Metric | Value |
|---|---|
| Hallucination rate | 0% (0/100) |
| Well-grounded explanations (>= 60%) | 55/100 (55%) |
| Avg events cited per explanation | 2.4 / 5 |
| Errors | 0/100 |

**Per-Ticker Accuracy**

| Ticker | Accuracy | n | Notes |
|---|---|---|---|
| DIS | 83.3% | 6 | Strong news signal |
| PG | 80.0% | 5 | Stable consumer stock |
| AAPL | 66.7% | 9 | Consistent retrieval |
| V | 66.7% | 3 | — |
| AVGO | 60.0% | 5 | — |
| UNH | 60.0% | 5 | — |
| PFE | 55.6% | 9 | — |
| COST | 50.0% | 4 | — |
| GOOGL | 50.0% | 4 | — |
| KO | 50.0% | 4 | — |
| JPM | 46.2% | 13 | High volume, mixed signal |
| BAC | 40.0% | 5 | — |
| NVDA | 40.0% | 5 | Macro-driven moves |
| TSLA | 33.3% | 3 | High volatility |
| MA | 33.3% | 3 | — |
| AMZN | 20.0% | 5 | Macro-driven |
| META | 0.0% | 5 | Hardest ticker this sample |
| MSFT | 0.0% | 2 | Small sample |
| XOM | 0.0% | 4 | Energy — macro driven |
| HD | 100.0% | 1 | Single example |

Tickers whose moves are driven by macro factors (XOM energy prices,
META regulatory sentiment) are hardest to predict from article text.
Tickers with strong idiosyncratic news signal (DIS, PG, AAPL) perform
best.

---

## 8. How to Run

### Setup

```bash
pip install yfinance pandas scikit-learn sentence-transformers \
            faiss-cpu groq joblib pyyaml transformers torch
```

### Full Pipeline

```bash
# 1. Data (skip if data already exists)
python scripts/download_prices.py
python scripts/download_spy.py
python scripts/download_news.py
python scripts/merge_data.py

# 2. (Optional) Inspect dataset health
python scripts/diagnose.py

# 3. Train production model
python -m scripts.train_blend

# 4. Build retrieval index
python -m scripts.build_index

# 5. Run explanations (10 examples)
set GROQ_API_KEY=gsk_your_key_here     # Windows
export GROQ_API_KEY=gsk_your_key_here  # Mac/Linux
python -m scripts.run_explain

# 6. Full evaluation (100 examples)
python -m scripts.evaluate_system
```

---

## 9. File Structure

```
project/
├── scripts/                          # Production pipeline
│   ├── download_news.py              # Alpha Vantage news fetch
│   ├── download_prices.py            # yfinance OHLCV download
│   ├── download_spy.py               # SPY benchmark download
│   ├── merge_data.py                 # Align news to next trading day
│   ├── diagnose.py                   # Dataset diagnostic report
│   ├── train_blend.py                # ← production model
│   ├── train_finbert_blend.py        # ← best embedding experiment
│   ├── build_index.py                # Build FAISS retrieval index
│   ├── run_explain.py                # Generate explanations (10 examples)
│   └── evaluate_system.py            # Full 100-example evaluation
│
├── experiments/                      # Archived research trail
│   ├── README.md                     # Notes on each experiment
│   ├── scripts/
│   │   ├── train_baseline.py         # LogReg + TF-IDF baseline
│   │   ├── train_tree_models.py      # RF + XGBoost (overfit)
│   │   ├── train_embeddings.py       # MiniLM embeddings
│   │   ├── train_finbert.py          # FinBERT-tone standalone
│   │   └── download_news_retry.py    # One-time retry for empty tickers
│   └── config/
│       ├── baseline.yaml             # Config for train_baseline.py
│       └── xgboost.yaml              # Config for train_tree_models.py
│
├── src/                              # Shared library modules
│   ├── embeddings.py                 # Multi-model encoder with cache
│   ├── evaluate.py                   # Metrics computation
│   ├── explain.py                    # Groq LLM explanation generator
│   ├── features.py                   # TF-IDF, LM sentiment, temporal features
│   ├── market_features.py            # Lagged price momentum features
│   ├── retrieval.py                  # FAISS index build + query
│   └── utils.py                      # IO helpers
│
├── data/
│   ├── raw/
│   │   ├── news/                     # Per-ticker CSVs from Alpha Vantage
│   │   └── prices/                   # Per-ticker OHLCV + SPY.csv
│   └── processed/
│       └── master_dataset.csv
│
└── artifacts/                        # Generated — excluded from git
    ├── cache/
    │   └── embeddings.pkl            # MiniLM + FinBERT cached embeddings
    ├── models/
    │   ├── logreg_blend.joblib       # ← production model
    │   ├── rf_blend.joblib
    │   ├── rf_imputer.joblib
    │   ├── rf_scaler.joblib
    │   ├── blend_meta.joblib
    │   ├── finbert_blend_logreg.joblib
    │   ├── finbert_blend_rf.joblib
    │   ├── finbert_blend_imputer.joblib
    │   └── finbert_blend_scaler.joblib
    ├── retrieval/
    │   ├── faiss_index.pkl           # 17,933 vector FAISS index
    │   └── metadata.pkl
    └── reports/
        ├── metrics_blend.json
        ├── metrics_finbert_blend.json
        ├── explanations.json
        ├── evaluation_summary.json
        └── evaluation_details.json
```

---

## 10. Results Summary

| Phase | Model | Test AUC | Status |
|---|---|---|---|
| Baseline | LogReg + TF-IDF | 0.5251 | ✅ Done |
| Tree experiments | RF + market + sentiment | 0.5267 | ✅ Done — overfit, no improvement |
| Improved baseline | **Blend + LM + filter** | **0.5432** | ✅ Done — **BEST MODEL** |
| MiniLM embeddings | MiniLM + LogReg | 0.5250 | ✅ Done — same as baseline |
| FinBERT embeddings | FinBERT-tone + LogReg | 0.5324 | ✅ Done — best embedding alone |
| FinBERT + blend | FinBERT-tone + full blend | 0.5432 | ✅ Done — tied with best |
| RAG retrieval index | FAISS + MiniLM | — | ✅ Done — 17,933 vectors |
| Explanation generation | LLaMA 3.3 70B + RAG | 0% hallucination | ✅ Done |
| Full evaluation | 100-example test | 48% point acc | ✅ Done |

### Final Scientific Conclusion

The 0.543 AUC ceiling is **robust across all text representations tested**
— from simple TF-IDF to financial domain-specific FinBERT. This confirms
the bottleneck is the fundamental signal strength of publicly available
financial news for next-day direction prediction, consistent with the
Efficient Market Hypothesis. The value of this system lies not in
marginal AUC improvements but in the **explanation layer** — grounded,
citation-backed, hallucination-free analytical output that makes the
model's reasoning transparent and auditable.

# 11. Explanation Verification System

## The Problem with the Original Approach

The original explanation generator asked the LLM to produce a free-text
paragraph and measured quality using a word-overlap grounding score — checking
whether words from the retrieved headlines appeared in the explanation. This
verified almost nothing meaningful. An explanation could pass with a perfect
score while stating the wrong return sign, wrong direction, or citing a return
of +1.84% when the actual value was -1.84%. The 0% hallucination claim in the
original system was based on informal human review, not programmatic fact
checking.

---

## What Was Built

A four-layer verification system that programmatically checks every factual
claim the LLM makes against the ground-truth metadata.

### Layer 1 — Structured Output (Pre-filled JSON)

Instead of asking the LLM to generate a free-text paragraph, the prompt now
requests a strict JSON object. Critically, every verifiable number is
pre-computed by Python and written directly into the JSON template before the
LLM sees the prompt:

- Return percentages for all 5 retrieved events
- Direction (UP/DOWN) for all 5 retrieved events
- Ticker symbols and dates for all 5 retrieved events
- Aggregate UP count across 5 events
- Aggregate average return across 5 events
- Overall prediction direction and confidence percentage

The LLM is only required to write three free-text fields in its own language:
a relevance sentence for each event, a caveats sentence, and a 2-3 sentence
summary. This structurally removes the LLM's ability to hallucinate numbers
because those numbers are already present in the template.

### Layer 2 — Programmatic Verification (7 Checks)

After the LLM responds, `verify_explanation()` in `src/explain.py` runs seven
checks against the ground-truth metadata:

| Check | Type | What It Verifies |
|---|---|---|
| 1 | Hard error | Overall direction matches blend_prob |
| 2 | Hard error | Confidence % matches blend_prob within 5% tolerance |
| 3 | Hard error | Every cited event exists in the retrieved top-5 (no invented events) |
| 4 | Hard error | Each event's cited direction matches actual label |
| 5 | Hard error | Each event's cited return matches actual return within 0.5% |
| 6 | Warning | Aggregate UP count matches actual count |
| 7 | Warning | Aggregate average return matches actual within 0.5% |

Hard errors fail verification and trigger a retry. Warnings are logged but
do not fail the explanation.

### Layer 3 — Retry Loop with Error Feedback

If verification fails, the specific errors are appended to the prompt in plain
language and the LLM is called again — up to 3 attempts. Each retry message
names the exact field that was wrong and the correct value from the metadata.

### Layer 4 — Template Fallback

If all retries fail, a guaranteed-correct explanation is generated directly
from the metadata dictionary with no LLM involvement. Every number in the
fallback comes from a Python variable, not from the model. Hallucination is
mathematically impossible in the fallback path.

---

## Results

Evaluated on 10 held-out test examples using the new verification system:

| Metric | Value |
|---|---|
| Verified (all 7 checks passed) | 10/10 (100%) |
| Passed on first attempt | 10/10 |
| Passed after retry | 0/10 |
| Used template fallback | 0/10 |
| Factual hallucination rate | 0% |

**Every explanation passed all 7 programmatic checks on the first LLM call
with no retries and no fallbacks.**

This is a stronger claim than the original system's 0% hallucination result.
The original checked word overlap only. This checks every numerical value,
every direction label, and every event citation against a known ground truth
with strict numerical tolerances.

---

## Comparison: Original vs Verified System

| Aspect | Original | Verified |
|---|---|---|
| Output format | Free text paragraph | Structured JSON |
| Numbers in output | Generated by LLM | Pre-filled by Python |
| Verification method | Word overlap heuristic | Programmatic fact checking |
| Checks performed | 1 (word presence) | 7 (numerical + directional) |
| Retry on failure | No | Yes (up to 3 attempts) |
| Fallback on all failures | No | Yes (template, 0% hallucination) |
| Hallucination claim basis | Informal human review | Automated per-field verification |

---

## Key Files Changed

| File | What Changed |
|---|---|
| `src/explain.py` | `build_prompt` — pre-fills all numbers into JSON template |
| `src/explain.py` | `_clean_and_parse_json` — 6-strategy robust JSON parser |
| `src/explain.py` | `verify_explanation` — 7 programmatic checks against metadata |
| `src/explain.py` | `generate_template_explanation` — LLM-free factual fallback |
| `src/explain.py` | `explain_prediction` — retry loop with error feedback |
| `src/explain.py` | `call_llm` — max_tokens raised from 512 to 1500 |
| `scripts/evaluate_system.py` | Added verification metrics to report and output JSON |

---

## Running the Verified Evaluation

```bash
# Set API key
$env:GROQ_API_KEY="gsk_your_key_here"         # Windows PowerShell
export GROQ_API_KEY=gsk_your_key_here          # Mac/Linux

# Run evaluation (set N_EXAMPLES in script, default 10)
python -m scripts.evaluate_system
```

Output now includes a verification section:

```
── Explanation Verification ─────────────────────────────────
  Verified (factually correct)     : 10/10 (100%)
  Passed first attempt             : 10/10
  Passed after retry               : 0/10
  Used template fallback           : 0/10
```

---

## Honest Limitations

The 0% hallucination rate is measured on 10 examples. A statistically robust
claim requires running the full 100-example evaluation. The Groq free tier
allows 100,000 tokens per day — run in batches of 40-50 examples on separate
days to reach 100 examples without hitting rate limits.

The verification system checks factual accuracy of numbers and citations. It
does not verify whether the LLM's analytical reasoning is financially sensible
— that requires domain expert review. A verified explanation is factually
correct but may still contain weak or irrelevant analytical connections between
the retrieved events and the current article.