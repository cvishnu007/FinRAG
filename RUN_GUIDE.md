# FinRAG: End-to-End Setup & Execution Guide

This guide explains how to clone, set up, and run the FinRAG pipeline from scratch.

## ⚠️ Known Limitations & Issues Before You Start

Before running the pipeline, please be aware of the following required steps and potential bottlenecks:

1. **Alpha Vantage API Rate Limits (Data Bottleneck):**
   - The `download_news.py` script attempts 40 API calls (20 tickers × 2 time windows).
   - Alpha Vantage's free tier is strictly limited to **25 calls per day**.
   - **Result:** The script will fail halfway through if using a free key. You will need to either run it across two consecutive days or use a premium API key.
2. **Missing Local Artifacts:**
   - Because of GitHub file size limits, the trained models (`.joblib`), vector indices (`faiss_index.pkl`), and raw data (`.csv`) are intentionally ignored via `.gitignore`. You **must** run the data ingestion and training scripts locally before you can evaluate the system.
3. **Groq API Key Required:**
   - The explanation generation step relies on the Groq LLM API. You must create a free Groq account, generate an API key, and export it to your environment.

---

## Step-by-Step Execution

### 1. Clone the Repository
```bash
git clone https://github.com/cvishnu007/FinRAG.git
cd FinRAG
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Set your Groq API key to enable the LLM explanation generator.
- **Mac/Linux:** `export GROQ_API_KEY="gsk_your_key_here"`
- **Windows (CMD):** `set GROQ_API_KEY="gsk_your_key_here"`
- **Windows (PowerShell):** `$env:GROQ_API_KEY="gsk_your_key_here"`

### 4. Run Data Ingestion
*(Note: Refer to Limitation #1 regarding API rate limits during the news download step).*
```bash
python scripts/download_prices.py
python scripts/download_spy.py
python scripts/download_news.py
python scripts/merge_data.py
```
*(Optional) Check dataset health:*
```bash
python scripts/diagnose.py
```

### 5. Train Models & Build Retrieval Index
Train the ensemble model and build the FAISS vector database for semantic search. This will create the required files in your local `artifacts/` folder.
```bash
python -m scripts.train_blend
python -m scripts.build_index
```

### 6. Generate Explanations & Evaluate
Run the RAG system to predict stock movements and explain the reasoning using historical data.
```bash
# Run a quick 10-example test
python -m scripts.run_explain

# Run the full 100-example evaluation pipeline
python -m scripts.evaluate_system
```

## What else might cause issues during execution?
- Hugging Face Model Downloads: The first time you run the training or index-building scripts, sentence-transformers will download the all-MiniLM-L6-v2 and FinBERT-tone models. If the user has a slow connection or Hugging Face is experiencing downtime, the script might hang or timeout.
- yfinance Blocking: Yahoo Finance (yfinance) occasionally throttles or blocks IPs that request too much data too quickly. While your current script volume is low, users running on shared IP addresses (like a VPN) might experience connection resets.