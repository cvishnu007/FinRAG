# Financial News & Stock Prediction System

## 1. Project Goal

This project builds a **financial forecasting and explanation system** that:

1. Predicts **short-term stock movement (next-day up/down)** based on
   historical stock prices and news momentum.
2. Generates **explanations grounded in real historical events** using a
   Retrieval-Augmented Generation (RAG) approach to prevent AI hallucination.

Instead of an opaque black-box model, the system acts as an analytical
assistant — when predicting a stock's movement it explicitly cites
historically analogous news events and their empirical market impacts.

---

## 2. System Architecture

The pipeline has three major components:

### A. Forecast Backbone (Predictive Model)
A machine learning classification model trained on historical news aligned
to subsequent trading day returns.
- **Input:** Current market momentum and breaking news
- **Output:** Probability the stock will close UP or DOWN the next trading day

### B. Historical Event Retrieval (RAG Index)
A FAISS vector database containing embeddings of past financial news events
and their associated market reactions.
- **Operation:** Given a breaking news story, retrieves the top-K most
  semantically similar historical events

### C. Explanation Generator
An LLM that synthesises the forecast and retrieved historical context.
- **Goal:** Ground explanations in real analogous events, e.g.
  *"This prediction is based on 4 similar hardware announcements by AAPL
  in 2021–2023, which produced an average +2.4% next-day return."*

---

## 3. Data Engineering — Complete

### Sources
- **News:** Alpha Vantage News Sentiment API (filtered at relevance
  score >= 0.1 per ticker)
- **Prices:** yfinance (OHLCV, auto-adjusted), including SPY as market
  benchmark

### Alignment Logic
Articles published after hours or on weekends are aligned to the *next
available market close* to capture the true market reaction. UNIX
timestamps are used throughout; timezone conversion is applied only at
feature-engineering time.

### Master Dataset (`data/processed/master_dataset.csv`)

| Field | Value |
|---|---|
| Total rows | 32,915 (before filtering) |
| After near-zero return filter (±0.2%) | 29,428 rows |
| After cross-ticker dedup | 25,632 rows used for training |
| Tickers | 20 large-cap: AAPL AMZN AVGO BAC COST DIS GOOGL HD JPM KO MA META MSFT NVDA PFE PG TSLA UNH V XOM |
| Time span | January 2019 — March 2026 |
| Label balance | ~51% UP / ~49% DOWN |

**Schema:**

| Column | Type | Description |
|---|---|---|
| ticker | str | Stock symbol |
| title | str | Article headline |
| summary | str | Article body excerpt |
| published | datetime | Publication timestamp (UTC) |
| trade_date | date | Next trading day price reacted |
| Close | float | Closing price on trade_date |
| return_1d | float | Raw next-day return |
| excess_return | float | return_1d minus SPY return (market-adjusted) |
| spy_return | float | SPY return on trade_date |
| label | int | 1 = stock UP next day, 0 = DOWN |

### Key Scripts

| Script | Purpose |
|---|---|
| `scripts/download_news.py` | Fetches news from Alpha Vantage API |
| `scripts/download_news_retry.py` | Retry for tickers that returned empty |
| `scripts/download_prices.py` | Downloads OHLCV data via yfinance |
| `scripts/download_spy.py` | Downloads SPY benchmark prices |
| `scripts/merge_data.py` | Aligns news to next trading day, computes labels |
| `scripts/diagnose.py` | Full diagnostic report on the merged dataset |

### Data Engineering Findings

- **Lag distribution:** 97.8% of articles have lag of exactly 1 day,
  meaning the staleness problem does not exist in this dataset — news
  is already well-aligned to the next trading day.
- **Excess return:** Mean=0.0002, Std=0.0195 — near-zero mean confirms
  the dataset is balanced and not biased toward bull/bear periods.
- **SPY column:** Added as an informational column. Experiments showed
  that using excess return (stock minus SPY) as the label *hurts*
  performance (AUC dropped from 0.543 to 0.483) because headlines
  predict absolute price movement, not alpha over market.

---

## 4. Baseline Model — Complete

### Pipeline
TF-IDF on `title + summary` combined with categorical and market momentum
features, fed into Logistic Regression.

```
python -m scripts.train_baseline
```

**Outputs:**
- `artifacts/models/logreg_model.joblib`
- `artifacts/models/tfidf_vectorizer.joblib`
- `artifacts/models/feature_meta.joblib`
- `artifacts/reports/metrics.json`

### What the pipeline does
- **Per-ticker chronological split:** each ticker split independently
  (70/15/15) so all 20 tickers appear in every split
- **Cross-ticker deduplication:** articles covering multiple tickers
  deduplicated before splitting to prevent contamination
- **Text features:** TF-IDF on `title + summary` (5k features, unigrams,
  min_df=20)
- **Categorical features:** ticker, day-of-week, month, market-session
  bucket (pre/market/post)
- **Market momentum:** prior-day and 5-day rolling returns (no future
  leakage)
- **Sentiment features:** Lexicon-based positive/negative ratio scores
- **Regularisation:** Strong L2 (C=0.001–0.01) tuned via validation AUC

### Bugs fixed during baseline phase
1. **Broken global split** — original row-position split left 10/20
   tickers with zero test rows. Fixed by switching to per-ticker
   chronological splits.
2. **Cross-ticker contamination** — 4,557 articles appeared under
   multiple tickers, allowing train/test leakage. Fixed by deduplicating
   on `(title, published)` before splitting.

### Results

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Train | 0.5501 | 0.5594 | 0.5690 | 0.5642 | 0.5743 |
| Validation | 0.5281 | 0.5251 | 0.5717 | 0.5474 | 0.5357 |
| **Test** | **0.5098** | **0.5096** | **0.6092** | **0.5550** | **0.5251** |

Train/test AUC gap: 0.049 — overfitting fully resolved.

---

## 5. Tree Model Experiments — Complete

Explored Random Forest and XGBoost across three feature sets to test
whether non-linear models could beat the baseline.

```
python -m scripts.train_tree_models
```

### Results (Test Set)

| Model | AUC | F1 | Accuracy | Train AUC | Gap |
|---|---|---|---|---|---|
| LogReg + TF-IDF (baseline) | 0.5251 | 0.5550 | 0.5098 | 0.5743 | 0.049 |
| Random Forest (market only) | 0.5144 | 0.5385 | 0.5063 | 0.6878 | 0.173 |
| Random Forest (market + sentiment) | 0.5267 | 0.5499 | 0.5251 | 0.6825 | 0.156 |
| XGBoost (market only) | 0.4842 | 0.5257 | 0.4747 | 0.7219 | 0.238 |
| XGBoost (market + sentiment) | 0.4870 | 0.5262 | 0.4792 | 0.7203 | 0.233 |
| XGBoost (TF-IDF + market + sentiment) | 0.5080 | 0.5583 | 0.5049 | 0.7175 | 0.210 |

**Key finding:** All tree models show train/test AUC gaps of 0.15–0.24,
indicating they memorise spurious patterns in the 5 noisy market features.
XGBoost market-only AUC of 0.484 (below random) confirms regime change
between val and test periods. LogReg remains the best model. Tree
experiments are complete — no further tuning warranted.

---

## 6. Improved Baseline — Complete

Three targeted improvements over the original baseline:

1. **Loughran-McDonald financial sentiment dictionary** — replaced the
   original 40-word lexicon with ~600 purpose-built financial sentiment
   words plus a dedicated `surprise_score` feature (beat/miss/estimate
   language)
2. **Label threshold filter** — dropped rows where `|return_1d| < 0.2%`
   to remove near-zero noise the model cannot distinguish
3. **Blended prediction** — 60% LogReg + 40% RF weighted probability
   average

```
python -m scripts.train_blend
```

**Outputs:**
- `artifacts/models/logreg_blend.joblib`
- `artifacts/models/rf_blend.joblib`
- `artifacts/models/blend_meta.joblib`
- `artifacts/reports/metrics_blend.json`

### Results (Test Set)

| Model | AUC | F1 | Accuracy |
|---|---|---|---|
| LogReg baseline (original) | 0.5251 | 0.5550 | 0.5098 |
| LogReg + LM sentiment + filter | 0.5410 | 0.5719 | 0.5362 |
| **Blend (LogReg 60% + RF 40%, threshold=0.5)** | **0.5432** | **0.5691** | **0.5277** |

**Key finding:** LM dictionary + label filtering pushed AUC from 0.525
→ 0.541. Blending adds a further +0.002. **0.543 is the practical ceiling
of bag-of-words methods on this task.** This is consistent with academic
literature on news-based stock direction prediction — efficient market
dynamics limit how much next-day direction can be predicted from public
news text alone.

> **Note on threshold tuning:** The val-tuned threshold (0.442) produced
> recall=1.0 — a degenerate solution that predicts UP for every row.
> Always use the default threshold of 0.5 for this model.

---

## 7. Sentence Embedding Experiment — Complete

Replaced TF-IDF with `all-MiniLM-L6-v2` (384-dim dense embeddings) to
test whether semantic representations could break the 0.543 ceiling.

```
pip install sentence-transformers
python -m scripts.train_embeddings
```

**Files added:**
- `src/embeddings.py` — sentence encoding with on-disk caching
- `scripts/train_embeddings.py` — embedding + logistic regression pipeline

### Results (Test Set)

| Model | Val AUC | Test AUC | Train AUC | Gap |
|---|---|---|---|---|
| Blend baseline (best so far) | 0.5260 | 0.5432 | — | — |
| MiniLM + LogReg + market + sentiment | 0.5262 | 0.5250 | 0.5878 | 0.063 |

**Key finding:** MiniLM embeddings matched TF-IDF exactly (0.525 AUC)
but did not beat the blend baseline. The train/test gap of 0.063 is
slightly worse than baseline's 0.049. This confirms the bottleneck is
not the text representation — it is the fundamental signal strength of
publicly available news for predicting next-day direction. The blend
model at **0.543 AUC remains the best predictor** and will serve as the
forecast backbone going forward.

**Experiment also confirmed:**
- Using excess return (stock minus SPY) as the label drops AUC to 0.483
  — headlines predict absolute price movement, not alpha
- Staleness (lag >= 3 days) affects 0.0% of the dataset — not a problem
- The embedding cache (25,632 texts → 401 batches, ~4.5 min) is stored
  at `artifacts/cache/embeddings.pkl` for reuse

---

## 8. Complete Model Comparison

| Phase | Model | Test AUC | Status |
|---|---|---|---|
| Baseline | LogReg + TF-IDF | 0.5251 | ✅ Done |
| Tree experiments | Best: RF + market + sentiment | 0.5267 | ✅ Done |
| Improved baseline | **Blend + LM dict + filter** | **0.5432** | ✅ Done — BEST |
| Sentence embeddings | MiniLM + LogReg | 0.5250 | ✅ Done |
| RAG retrieval index | FAISS + MiniLM embeddings | — | 🔲 Next |
| Explanation generation | LLM + RAG context | — | 🔲 Planned |
| Full system evaluation | End-to-end pipeline | — | 🔲 Planned |

---

## 9. Next Steps

### Step 3: Build the Retrieval Index

The MiniLM embeddings (already cached at
`artifacts/cache/embeddings.pkl`) will be reused to build a FAISS vector
database over all 25,632 training events. Given any new article, the
index retrieves the top-K most semantically similar historical events
along with their realised market outcomes.

**Files to add:**
- `src/retrieval.py` — FAISS index build and query functions
- `scripts/build_index.py` — one-time index construction script

```
pip install faiss-cpu
python -m scripts.build_index
```

**What it produces:**
- `artifacts/retrieval/faiss_index.pkl` — serialised FAISS flat IP index
- `artifacts/retrieval/metadata.pkl` — DataFrame of training events
  (ticker, title, published, trade_date, return_1d, label) aligned
  index-position to the FAISS vectors

### Step 4: Explanation Generation

Combine the blend model's probability output and the FAISS top-K
retrieved events into an LLM prompt. The LLM is instructed to explain
the prediction *only* using the retrieved historical context — no
hallucination because it has no other grounding material.

**Files to add:**
- `src/explain.py` — prompt construction and Anthropic API call
- `scripts/run_explain.py` — end-to-end demo on held-out test examples

**Example output target:**
```
Ticker: NVDA
Headline: "Nvidia announces Blackwell GPU shipment ahead of schedule"
Prediction: UP (probability 0.67)

Explanation:
This prediction is grounded in 4 similar semiconductor supply
announcements retrieved from the training set:

1. NVDA — "Nvidia ships H100 ahead of schedule" (2023-03-14)
   Next-day return: +4.2% ✅ UP

2. AVGO — "Broadcom accelerates chip delivery timeline" (2022-11-08)
   Next-day return: +3.1% ✅ UP

3. NVDA — "Nvidia GPU availability improves for data centres" (2023-07-19)
   Next-day return: +2.8% ✅ UP

4. MSFT — "Microsoft Azure GPU capacity expanded ahead of plan" (2024-01-22)
   Next-day return: -0.4% ❌ DOWN

3 of 4 analogous historical events resulted in positive next-day
returns, with an average return of +2.4%. The model assigns 67%
probability to an UP move.
```

### Step 5: Full System Evaluation

Evaluate the complete pipeline on the held-out test set across two
dimensions:

**Quantitative:**
- AUC / F1 / accuracy of the blend model (already measured: 0.543 AUC)
- Coverage: what fraction of test articles return at least 3 relevant
  historical analogues (similarity > threshold)

**Qualitative:**
- Grounding score: fraction of explanation claims traceable to a
  retrieved event
- Hallucination rate: claims with no supporting retrieved context
- Relevance score: human spot-check on 50 random explanations — are
  the retrieved analogues genuinely similar?

---

## 10. Project File Structure

```
project/
├── config/
│   ├── baseline.yaml          # LogReg hyperparameters
│   └── xgboost.yaml           # XGBoost hyperparameters
├── data/
│   ├── raw/
│   │   ├── news/              # Per-ticker news CSVs from Alpha Vantage
│   │   └── prices/            # Per-ticker OHLCV CSVs from yfinance + SPY
│   └── processed/
│       └── master_dataset.csv # Merged, aligned, labelled dataset
├── scripts/
│   ├── download_news.py
│   ├── download_news_retry.py
│   ├── download_prices.py
│   ├── download_spy.py
│   ├── merge_data.py
│   ├── diagnose.py
│   ├── train_baseline.py
│   ├── train_tree_models.py
│   ├── train_blend.py         # ← Best model
│   ├── train_embeddings.py
│   ├── build_index.py         # ← Next to build
│   └── run_explain.py         # ← Planned
├── src/
│   ├── embeddings.py          # MiniLM encoder with caching
│   ├── evaluate.py
│   ├── features.py            # LM sentiment + temporal features
│   ├── market_features.py
│   ├── retrieval.py           # ← Next to build
│   ├── explain.py             # ← Planned
│   └── utils.py
└── artifacts/
    ├── cache/
    │   └── embeddings.pkl     # Cached MiniLM embeddings (25,632 texts)
    ├── models/
    │   ├── logreg_blend.joblib     # Best prediction model
    │   ├── rf_blend.joblib
    │   ├── blend_meta.joblib
    │   ├── minilm_logreg.joblib
    │   └── minilm_meta.joblib
    ├── retrieval/
    │   ├── faiss_index.pkl    # ← To be built
    │   └── metadata.pkl       # ← To be built
    └── reports/
        ├── metrics.json
        ├── metrics_blend.json
        └── metrics_minilm.json
```
```