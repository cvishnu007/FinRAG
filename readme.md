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

## 3. Data Engineering — Complete

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

### Key Scripts

| Script | Purpose |
|---|---|
| `scripts/download_news.py` | Alpha Vantage news fetch |
| `scripts/download_news_retry.py` | Retry for empty tickers |
| `scripts/download_prices.py` | yfinance OHLCV download |
| `scripts/download_spy.py` | SPY benchmark download |
| `scripts/merge_data.py` | Align news to next trading day |
| `scripts/diagnose.py` | Dataset diagnostic report |

---

## 4. Modelling — Complete

### All Experiments

| Phase | Model | Test AUC | Train AUC | Gap | Notes |
|---|---|---|---|---|---|
| Baseline | LogReg + TF-IDF | 0.5251 | 0.5743 | 0.049 | Clean, no leakage |
| Trees | Random Forest (best) | 0.5267 | 0.6825 | 0.156 | Overfit |
| Trees | XGBoost (best) | 0.5080 | 0.7175 | 0.210 | Severe overfit |
| Improved | LogReg + LM sentiment | 0.5410 | — | — | LM dict key win |
| **Best** | **Blend + LM + filter** | **0.5432** | — | — | **Production** |
| Embeddings | MiniLM + LogReg | 0.5250 | 0.5878 | 0.063 | Same as TF-IDF |

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

**Why AUC is ~0.54:** Consistent with academic literature on news-based
stock direction prediction. Efficient market hypothesis ensures publicly
available news is largely priced in by open. The 0.543 ceiling is the
honest limit of bag-of-words methods on this task.

**Artifacts:**
artifacts/models/logreg_blend.joblib
artifacts/models/rf_blend.joblib
artifacts/models/rf_imputer.joblib
artifacts/models/rf_scaler.joblib
artifacts/models/blend_meta.joblib
---

## 5. Sentence Embedding Experiment — Complete

Replaced TF-IDF with `all-MiniLM-L6-v2` (384-dim dense embeddings).

**Result:** Test AUC 0.5250 — identical to TF-IDF baseline. The
bottleneck is signal strength of public news, not the text
representation. The embedding cache is reused for FAISS retrieval.
artifacts/cache/embeddings.pkl    ← 25,632 cached MiniLM embeddings
---

## 6. RAG Retrieval Index — Complete

FAISS flat inner-product index built over 17,933 training-set embeddings.

```bash
python -m scripts.build_index
```

**Index stats:**

| Metric | Value |
|---|---|
| Vectors indexed | 17,933 |
| Embedding dimension | 384 |
| Search type | Exact inner product (cosine sim) |
| Avg retrieval similarity | 0.593 |

**Retrieval correctness check (smoke test):** Given a PFE COVID vaccine
article, retrieved 4 other Pfizer vaccine articles with similarities
0.633–0.644. Semantically precise.
artifacts/retrieval/faiss_index.pkl
artifacts/retrieval/metadata.pkl
---

## 7. Explanation Generation — Complete

LLaMA 3.3 70B via Groq API generates grounded explanations from the
blend model probability + top-5 retrieved historical analogues.

```bash
set GROQ_API_KEY=gsk_your_key_here
python -m scripts.run_explain
```

### Example Output
Ticker  : AVGO
Headline: Teledyne FLIR OEM Launches Boson+ IQ Thermal Imaging Kit
Date    : 2025-09-09
Actual  : DOWN ❌ (-2.69%)
Predicted: DOWN (blend prob: 48.5%) ✅ CORRECT
Retrieved Analogues:

[AVGO] Qualcomm-built processor by Teledyne FLIR (2024-04-17)
-1.84% | DOWN | sim=0.689
[NVDA] Teledyne FLIR Tracking Software announced (2022-06-23)
-1.50% | DOWN | sim=0.684

Explanation:
The model predicts a downward movement in AVGO with probability 48.6%.
This is grounded in the Qualcomm-built Teledyne FLIR processor
announcement on 2024-04-17 which led to a -1.84% decline in AVGO,
and the Teledyne FLIR software event on 2022-06-23 which produced
-1.50% in NVDA. The aggregate shows 2/5 analogues were UP with
average return -0.57%.
---

## 8. Full System Evaluation — Complete

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
> variance of the 54.3% AUC model (±10% confidence interval at n=100).

**Accuracy by Return Magnitude**

| Return bucket | Accuracy | n |
|---|---|---|
| Small (< 1%) | 52.2% | 46 |
| Medium (1–3%) | 46.7% | 45 |
| Large (> 3%) | 33.3% | 9 |

Large moves are hardest to predict — these are typically surprise events
where the market reaction is driven by factors beyond the text alone.

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

The LLM never fabricated events or cited sources outside the retrieved
context across all 100 examples.

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
| META | 0.0% | 5 | Hardest ticker |
| MSFT | 0.0% | 2 | Small sample |
| XOM | 0.0% | 4 | Energy macro driven |
| HD | 100.0% | 1 | Single example |

**Key finding:** Tickers whose moves are driven primarily by macro
factors (XOM energy prices, META regulatory sentiment, AMZN consumer
spending) are hardest to predict from individual article text. Tickers
with strong idiosyncratic news signal (DIS, PG, AAPL) perform best.

---

## 9. How to Run

### Setup

```bash
pip install yfinance pandas scikit-learn sentence-transformers \
            faiss-cpu groq joblib pyyaml xgboost anthropic
```

### Full pipeline

```bash
# Data (already done — skip if data exists)
python scripts/download_prices.py
python scripts/download_spy.py
python scripts/download_news.py
python scripts/merge_data.py

# Train best model
python -m scripts.train_blend

# Build retrieval index
python -m scripts.build_index

# Run explanations (10 examples)
set GROQ_API_KEY=gsk_your_key_here
python -m scripts.run_explain

# Full evaluation (100 examples)
python -m scripts.evaluate_system
```

---

## 10. File Structure
finRAG/
├── config/
│   ├── baseline.yaml
│   └── xgboost.yaml
├── data/
│   ├── raw/
│   │   ├── news/               # Per-ticker CSVs from Alpha Vantage
│   │   └── prices/             # Per-ticker OHLCV + SPY.csv
│   └── processed/
│       └── master_dataset.csv
├── scripts/
│   ├── download_news.py
│   ├── download_news_retry.py
│   ├── download_prices.py
│   ├── download_spy.py
│   ├── merge_data.py
│   ├── diagnose.py
│   ├── train_baseline.py
│   ├── train_tree_models.py
│   ├── train_blend.py          ← best model
│   ├── train_embeddings.py
│   ├── build_index.py
│   ├── run_explain.py
│   └── evaluate_system.py
├── src/
│   ├── embeddings.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── features.py
│   ├── market_features.py
│   ├── retrieval.py
│   └── utils.py
└── artifacts/
├── cache/
│   └── embeddings.pkl        # 25,632 MiniLM embeddings
├── models/
│   ├── logreg_blend.joblib   # production model
│   ├── rf_blend.joblib
│   ├── rf_imputer.joblib
│   ├── rf_scaler.joblib
│   └── blend_meta.joblib
├── retrieval/
│   ├── faiss_index.pkl       # 17,933 vector FAISS index
│   └── metadata.pkl
└── reports/
├── metrics.json
├── metrics_blend.json
├── metrics_minilm.json
├── explanations.json
├── evaluation_summary.json
└── evaluation_details.json
---

## 11. Project Status — Complete

| Phase | Status | Key Result |
|---|---|---|
| Data engineering | ✅ Complete | 25,632 clean aligned events, 2019–2026 |
| Baseline model | ✅ Complete | AUC 0.525, no leakage |
| Tree experiments | ✅ Complete | No improvement — all overfit |
| Improved baseline | ✅ Complete | AUC 0.543 — best predictive model |
| Sentence embeddings | ✅ Complete | AUC 0.525 — confirmed text ceiling |
| RAG retrieval index | ✅ Complete | 17,933 vectors, avg sim 0.593 |
| Explanation generation | ✅ Complete | 0% hallucination, 100 examples |
| Full evaluation | ✅ Complete | 48% point acc, 55% well-grounded |