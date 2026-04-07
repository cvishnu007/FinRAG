# Financial News & Stock Prediction System

## 1. Project Goal

This project aims to build a **financial forecasting and explanation system** that:
1. Predicts **short-term stock movement (next-day up/down)** based on historical stock prices and news momentum.
2. Generates **explanations grounded in real historical events** using a Retrieval-Augmented Generation (RAG) approach to prevent AI hallucination.

Instead of an opaque "black-box" model, this system acts as an analytical assistant. When predicting a stock's movement, it explicitly cites historically analogous news events and their empirical market impacts.

---

## 2. System Architecture

The pipeline consists of three major components:

### A. Forecast Backbone (Predictive Model)
A machine learning classification model trained on the alignment of historical news and subsequent trading day returns.
* **Input:** Current market momentum and breaking news.
* **Output:** Probability the stock will close UP or DOWN the next day.

### B. Historical Event Retrieval (RAG Index)
A vector database (e.g., FAISS) containing embeddings of tens of thousands of past financial news events and their associated stock market reactions.
* **Operation:** Given a breaking news story, retrieves the top-K most semantically similar historical events.

### C. Explanation Generator
An LLM (Large Language Model) that synthesizes the forecast and the retrieved historical contexts.
* **Goal:** Generate a grounded explanation like:
  > *"This prediction is grounded in 4 similar hardware announcements by AAPL in 2021 and 2023, which resulted in an average 2.4% next-day price increase."*

---

## 3. Data Engineering Phase

The foundation of the project requires a perfectly aligned dataset of historical news and the subsequent market reaction. This phase is complete.

* **News Data:** Sourced via the Alpha Vantage News Sentiment API (filtered for high ticker relevance).
* **Price Data:** Sourced via `yfinance` (market Open/Close/High/Low/Volume).
* **Alignment Logic:** News articles published over weekends or after hours are meticulously aligned to the *next available market close* to capture the true market reaction. Date-time parsing and timezone removal ensure perfect synchronization.

### Final Master Dataset (`data/processed/master_dataset.csv`)
* **Total Rows:** 28,358 validated events (after removing 4,557 cross-ticker duplicate articles)
* **Tickers Covered:** 20 large-cap stocks — AAPL, AMZN, AVGO, BAC, COST, DIS, GOOGL, HD, JPM, KO, MA, META, MSFT, NVDA, PFE, PG, TSLA, UNH, V, XOM
* **Time Span:** January 2019 — December 2025
* **Label Balance:** 50.9% UP (1) / 49.1% DOWN (0)

**Data Format:**
| ticker | title                | summary | published (unix) | trade_date | Close | return_1d | label |
| ------ | -------------------- | ------- | ---------------- | ---------- | ----- | --------- | ----- |
| AAPL   | Apple beats earnings | ...     | 1690848000       | 2023-08-02 | 195.5 | +0.024    | 1     |

*(Note: `published` is the precise publication time in UNIX seconds. `trade_date` is the exact future market day the stock price reacted.)*

---

## 4. Baseline Model Phase (Complete)

### Pipeline Overview

The baseline pipeline uses TF-IDF on `title + summary` combined with categorical
and market momentum features, fed into a Logistic Regression classifier.

**Key files:**
- `scripts/train_baseline.py` — end-to-end training script
- `config/baseline.yaml` — feature flags, split ratios, model hyperparameters
- `src/features.py` — text, temporal, and sentiment feature construction
- `src/market_features.py` — lagged returns and rolling market features
- `src/evaluate.py` — metric computation and reporting
- `src/utils.py` — config loading and JSON output helpers

**How to run:**
```
python -m scripts.train_baseline
```

**Outputs:**
- `artifacts/models/logreg_model.joblib`
- `artifacts/models/tfidf_vectorizer.joblib`
- `artifacts/models/feature_meta.joblib`
- `artifacts/reports/metrics.json`

### What the pipeline does

- **Per-ticker chronological split:** each ticker is split independently (70/15/15)
  so all 20 tickers appear in every split, and chronological order is respected
  within each ticker
- **Cross-ticker deduplication:** articles covering multiple tickers are deduplicated
  before splitting to prevent train/test contamination
- **Text features:** TF-IDF on `title + summary` (5k features, unigrams, min_df=20)
- **Categorical features:** ticker, day-of-week, month, market-session bucket
- **Market momentum:** prior-day returns and 5-day rolling stats (no future leakage)
- **Sentiment features:** lexicon-based positive/negative ratio scores
- **Regularization:** strong L2 (C=0.001–0.01) tuned via validation AUC

### Bugs fixed during baseline phase

Two critical data bugs were identified and fixed before the final results below:

1. **Broken global split** — the original row-position split left 10 out of 20
   tickers with zero test rows (AAPL, AMZN, MSFT, GOOGL, TSLA, NVDA and others
   ran out of news before the test boundary). Fixed by switching to per-ticker
   chronological splits.

2. **Cross-ticker contamination** — 4,557 articles appeared under multiple tickers,
   meaning the same article could appear in both train and test under different
   ticker labels. Fixed by deduplicating on `(title, published)` before splitting.

### Final baseline results

| Split      | Accuracy | Precision | Recall | F1     | ROC-AUC |
|------------|----------|-----------|--------|--------|---------|
| Train      | 0.5501   | 0.5594    | 0.5690 | 0.5642 | 0.5743  |
| Validation | 0.5281   | 0.5251    | 0.5717 | 0.5474 | 0.5357  |
| Test       | 0.5098   | 0.5096    | 0.6092 | 0.5550 | 0.5251  |

**Key finding:** The train/test AUC gap is 0.05 (down from 0.34 before the data
fixes), confirming overfitting is fully resolved. The 0.525 test AUC represents
the true ceiling of TF-IDF + Logistic Regression on this task — consistent with
academic literature on news-based stock direction prediction. This establishes a
clean, honest baseline for all future models to beat.

---

## 5. Next Steps

### Step 2: Sentence Embedding Model
Replace TF-IDF with a pretrained sentence encoder (`all-MiniLM-L6-v2` via
`sentence-transformers`). Dense embeddings capture semantic meaning across time,
unlike TF-IDF which fails when vocabulary drifts between training and test
periods (e.g. "Blackwell", "ChatGPT", "omicron" dominate different periods and
don't generalise). Target: test AUC > 0.57.

**Files to add:**
- `src/embeddings.py` — sentence encoding with on-disk caching
- `scripts/train_embeddings.py` — embedding + logistic regression pipeline

**How to run (once implemented):**
```
python -m scripts.train_embeddings
```

### Step 3: Build the Retrieval Index (RAG)
Generate embeddings for all training-set events and ingest into a FAISS vector
database. Given a new news article, retrieve the top-K most semantically similar
historical events along with their realised market outcomes.

### Step 4: Implement RAG & Explanation Generation
Combine the classifier output and FAISS retrieval results into an LLM context
window. Write focused zero-shot or few-shot prompts that instruct the LLM to
explain the classifier's prediction *only* using the retrieved historical context,
preventing hallucination.

### Step 5: Full System Evaluation
Evaluate the complete pipeline on the held-out test set. Measure quantitative
predictive accuracy (AUC / F1) as well as qualitative grounding of the
LLM-generated explanations against hallucination metrics.