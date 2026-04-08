# Financial News & Stock Prediction System

## 1. Project Goal

This project builds a **financial forecasting and explanation system** that:

1. Predicts **short-term stock movement (next-day up/down)** based on historical stock prices and news momentum.
2. Generates **explanations grounded in real historical events** using a Retrieval-Augmented Generation (RAG) approach to prevent AI hallucination.

Instead of an opaque black-box model, the system acts as an analytical assistant — when predicting a stock's movement it explicitly cites historically analogous news events and their empirical market impacts.

---

## 2. System Architecture

The pipeline has three major components:

### A. Forecast Backbone (Predictive Model)
A machine learning classification model trained on historical news aligned to subsequent trading day returns.
- **Input:** Current market momentum and breaking news
- **Output:** Probability the stock will close UP or DOWN the next trading day

### B. Historical Event Retrieval (RAG Index)
A FAISS vector database containing embeddings of past financial news events and their associated market reactions.
- **Operation:** Given a breaking news story, retrieves the top-K most semantically similar historical events

### C. Explanation Generator
An LLM that synthesises the forecast and retrieved historical context.
- **Goal:** Ground explanations in real analogous events, e.g. *"This prediction is based on 4 similar hardware announcements by AAPL in 2021–2023, which produced an average +2.4% next-day return."*

---

## 3. Data Engineering Phase — Complete

### Sources
- **News:** Alpha Vantage News Sentiment API (filtered at relevance score >= 0.1 per ticker)
- **Prices:** yfinance (OHLCV, auto-adjusted)

### Alignment Logic
Articles published after hours or on weekends are aligned to the *next available market close* to capture the true market reaction. UNIX timestamps are used throughout; timezone conversion is applied only at feature-engineering time.

### Master Dataset (`data/processed/master_dataset.csv`)

| Field | Value |
|---|---|
| Total rows | 28,358 (after cross-ticker dedup) |
| Tickers | 20 large-cap: AAPL AMZN AVGO BAC COST DIS GOOGL HD JPM KO MA META MSFT NVDA PFE PG TSLA UNH V XOM |
| Time span | January 2019 — December 2025 |
| Label balance | 50.9% UP / 49.1% DOWN |

**Schema:**

| Column | Type | Description |
|---|---|---|
| ticker | str | Stock symbol |
| title | str | Article headline |
| summary | str | Article body excerpt |
| published | int | UNIX timestamp (seconds, UTC) |
| trade_date | date | Next trading day price reacted |
| Close | float | Closing price on trade_date |
| return_1d | float | (Close[t+1] - Close[t]) / Close[t] |
| label | int | 1 = UP, 0 = DOWN |

### Key Scripts

| Script | Purpose |
|---|---|
| `scripts/download_news.py` | Fetches news from Alpha Vantage API |
| `scripts/download_news_retry.py` | Retry script for tickers that returned empty |
| `scripts/download_prices.py` | Downloads OHLCV data via yfinance |
| `scripts/merge_data.py` | Aligns news to next trading day, computes labels |
| `scripts/diagnose.py` | Full diagnostic report on the merged dataset |

---

## 4. Baseline Model — Complete

### Pipeline

TF-IDF on `title + summary` combined with categorical and market momentum features, fed into Logistic Regression.

**Key files:**

| File | Purpose |
|---|---|
| `scripts/train_baseline.py` | End-to-end training script |
| `config/baseline.yaml` | Feature flags, split ratios, hyperparameters |
| `src/features.py` | Text, temporal, and sentiment feature construction |
| `src/market_features.py` | Lagged returns and rolling market features |
| `src/evaluate.py` | Metric computation and reporting |
| `src/utils.py` | Config loading and JSON output helpers |

```
python -m scripts.train_baseline
```

**Outputs:**
- `artifacts/models/logreg_model.joblib`
- `artifacts/models/tfidf_vectorizer.joblib`
- `artifacts/models/feature_meta.joblib`
- `artifacts/reports/metrics.json`

### What the pipeline does

- **Per-ticker chronological split:** each ticker split independently (70/15/15) so all 20 tickers appear in every split
- **Cross-ticker deduplication:** articles covering multiple tickers deduplicated before splitting to prevent contamination
- **Text features:** TF-IDF on `title + summary` (5k features, unigrams, min_df=20)
- **Categorical features:** ticker, day-of-week, month, market-session bucket (pre/market/post)
- **Market momentum:** prior-day and 5-day rolling returns (no future leakage)
- **Sentiment features:** Lexicon-based positive/negative ratio scores
- **Regularisation:** Strong L2 (C=0.001–0.01) tuned via validation AUC

### Bugs fixed

1. **Broken global split** — original row-position split left 10/20 tickers with zero test rows. Fixed by switching to per-ticker chronological splits.
2. **Cross-ticker contamination** — 4,557 articles appeared under multiple tickers, allowing train/test leakage. Fixed by deduplicating on `(title, published)` before splitting.

### Results

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Train | 0.5501 | 0.5594 | 0.5690 | 0.5642 | 0.5743 |
| Validation | 0.5281 | 0.5251 | 0.5717 | 0.5474 | 0.5357 |
| **Test** | **0.5098** | **0.5096** | **0.6092** | **0.5550** | **0.5251** |

Train/test AUC gap: 0.049 — overfitting fully resolved. This is the honest TF-IDF + LogReg ceiling, consistent with academic literature.

---

## 5. Tree Model Experiments — Complete

Explored Random Forest and XGBoost across three feature sets to test whether non-linear models could beat the baseline.

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

**Key finding:** All tree models show train/test gaps of 0.15–0.24, indicating they memorise spurious patterns in the 5 noisy market features. XGBoost market-only AUC of 0.484 (below random) is a sign of regime change between val and test periods. LogReg remains the best model. Tree experiments are complete — no further tuning warranted.

---

## 6. Improved Baseline — Complete

Three targeted improvements applied over the original baseline:

1. **Loughran-McDonald financial sentiment dictionary** — replaced the original 40-word lexicon with ~600 purpose-built financial sentiment words plus a dedicated `surprise_score` feature (beat/miss/estimate language)
2. **Label threshold filter** — dropped rows where `|return_1d| < 0.2%` to remove near-zero noise the model cannot predict
3. **Blended prediction** — 60% LogReg + 40% RF weighted probability average

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
| LogReg + LM sentiment + threshold filter | 0.5410 | 0.5719 | 0.5362 |
| **Blend (LogReg + RF, threshold=0.5)** | **0.5432** | **0.5691** | **0.5277** |

**Key finding:** LM dictionary + label filtering alone pushed AUC from 0.525 → 0.541. Blending adds a further marginal +0.002. The 0.543 AUC represents the practical ceiling of bag-of-words methods on this task. TF-IDF cannot distinguish "beats estimates" from "misses estimates" (3 shared tokens, 1 different); sentence embeddings can.

> Note: The optimal-threshold variant (val-tuned) produced recall=1.0 (predicts UP for every row) — a degenerate solution. Use the default 0.5 threshold. A precision-constrained threshold finder is implemented in the fixed version of `train_blend.py`.

---

## 7. Next Steps

### Step 2: Sentence Embedding Model

Replace TF-IDF with `all-MiniLM-L6-v2` via `sentence-transformers`. Dense 384-dim embeddings capture semantic meaning and generalise across vocabulary drift (e.g. "Blackwell", "ChatGPT", "omicron" appear in test but not train — TF-IDF gives them zero weight, MiniLM handles them through semantic similarity).

**Target:** test AUC > 0.57

**Files to add:**
- `src/embeddings.py` — sentence encoding with on-disk caching
- `scripts/train_embeddings.py` — embedding + logistic regression pipeline

```
pip install sentence-transformers
python -m scripts.train_embeddings
```

### Step 3: Build the Retrieval Index (RAG)

Generate embeddings for all training-set events and ingest into a FAISS vector database. Given a new article, retrieve the top-K most semantically similar historical events and their realised market outcomes.

**Files to add:**
- `src/retrieval.py` — FAISS index build and query
- `scripts/build_index.py` — one-time index construction

### Step 4: Explanation Generation

Combine classifier output and FAISS retrieval results into an LLM context window. Zero-shot prompts instruct the LLM to explain predictions *only* using retrieved historical context, preventing hallucination.

**Files to add:**
- `src/explain.py` — prompt construction and LLM call
- `scripts/run_explain.py` — end-to-end demo on test examples

### Step 5: Full System Evaluation

Evaluate the complete pipeline on the held-out test set:
- Quantitative: AUC / F1 / accuracy
- Qualitative: grounding score (what fraction of explanation claims are traceable to a retrieved event)
- Hallucination rate: claims with no supporting retrieved context

---

## 8. Model Comparison Summary

| Phase | Best Model | Test AUC | Status |
|---|---|---|---|
| Baseline | LogReg + TF-IDF | 0.525 | Done |
| Tree experiments | (none beat baseline) | 0.527 | Done |
| Improved baseline | Blend + LM dict + filter | 0.543 | Done |
| Sentence embeddings | all-MiniLM-L6-v2 | — | Next |
| RAG + explanations | Full system | — | Planned |