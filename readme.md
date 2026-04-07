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
* **Total Rows:** Over 20,000 strictly validated events
* **Tickers Covered:** 12 major large-cap stocks (AAPL, AMZN, GOOGL, HD, JPM, META, MSFT, NVDA, PG, TSLA, UNH, V). The remaining 8 target tickers are pending download.
* **Time Span:** January 2019 — March 2026
* **Label Balance:** 51.1% UP (1) / 48.9% DOWN (0)

**Data Format:**
| ticker | title                | summary | published (unix) | trade_date | Close | return_1d | label |
| ------ | -------------------- | ------- | ---------------- | ---------- | ----- | --------- | ----- |
| AAPL   | Apple beats earnings | ...     | 1690848000       | 2023-08-02 | 195.5 | +0.024    | 1     |

*(Note: `published` is the precise publication time in UNIX seconds. `trade_date` is the exact future market day the stock price reacted.)*

---

## 4. Current Status & Next Steps (Phase 2)

With the robust 20,000-row `master_dataset.csv` generated and totally free of class imbalance, the project is moving entirely out of Data Engineering and into the **Model Training & Evaluation Phase**.

## XGBoost Baseline (Chronological)

This adds an XGBoost classifier using the same feature engineering and
chronological split as the current ML pipeline.

### What it uses
- **Same split**: train oldest, validate next, test newest
- **Same features**: TF-IDF text + ticker + temporal + market momentum + sentiment
- **Optional SVD**: reduce TF-IDF dimensionality for speed and stability

### Run XGBoost
```
python -m scripts.train_xgboost
```

### Key files
- `scripts/train_xgboost.py` — XGBoost training with chrono split
- `config/xgboost.yaml` — hyperparameters and feature flags

### Step 2: Build the Retrieval Index
* **Task:** Generate dense text embeddings (e.g., via HuggingFace or OpenAI text-embedding models) for the `summary` or `title` of all events in the Training set.
* **Action:** Ingest these embeddings into a FAISS vector database instance.

### Step 3: Implement RAG & Explanation Generation
* **Task:** Combine the classifier output and the FAISS retrieval results into an LLM context window.
* **Action:** Write focused zero-shot or few-shot prompts that instruct the LLM to explain the classifier's prediction *only* using the retrieved historical context.

### Step 4: Full System Evaluation
* **Task:** Evaluate the full pipeline iteratively on the held-out Test set.
* **Action:** Measure quantitative predictive accuracy (AUC / F1 Score) as well as the qualitative grounding of the LLM-generated explanations against hallucination metrics.
