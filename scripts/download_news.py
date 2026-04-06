"""
download_news.py
================
Downloads financial news articles for each ticker using the
Alpha Vantage News Sentiment API (NEWS_SENTIMENT endpoint).

Strategy
--------
The free-tier limit is 25 API calls/day.  We have 20 tickers, so we fetch
each ticker separately.  Each call uses limit=1000 and sweeps from 2019-01-01
to today to maximise historical coverage.

Output
------
One CSV per ticker in  data/raw/news/<TICKER>.csv  with columns:
    ticker | title | summary | link | published   (UNIX timestamp, seconds)
"""

import requests
import pandas as pd
import os
import time

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = "76LSYWS76MP6V9NK"

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "NVDA", "META", "JPM",  "V",    "UNH",
    "HD",   "PG",   "MA",   "BAC",  "XOM",
    "AVGO", "PFE",  "KO",   "COST", "DIS",
]

# TICKERS = ["AVGO", "BAC", "COST", "DIS", "KO", "MA", "PFE", "XOM"]

# Date windows to sweep.  We split into two windows so we can call the API
# twice per ticker and collect articles from a wider time range while staying
# inside the limit=1000 cap per call.
TIME_WINDOWS = [
    ("20190101T0000", "20220101T0000"),   # 2019-01-01 → 2022-01-01
    ("20220101T0000", "20260314T0000"),   # 2022-01-01 → 2026-03-14 (today)
]

# Minimum relevance score for the article to be counted for this ticker.
# Alpha Vantage uses 0–1.  0.1 filters out very tangential mentions.
MIN_RELEVANCE = 0.1

BASE_URL = "https://www.alphavantage.co/query"

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "data", "raw", "news")
os.makedirs(output_dir, exist_ok=True)

# ── Helper ────────────────────────────────────────────────────────────────────

def fetch_news_window(ticker: str, time_from: str, time_to: str) -> list[dict]:
    """Fetch up to 1000 articles for *ticker* in the given time window."""
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   ticker,
        "time_from": time_from,
        "time_to":   time_to,
        "sort":      "EARLIEST",
        "limit":     1000,
        "apikey":    API_KEY,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for {ticker} [{time_from}–{time_to}]")
            return []
        data = resp.json()
    except Exception as e:
        print(f"    Request error for {ticker}: {e}")
        return []

    if "feed" not in data:
        # Could be a rate-limit message or error message
        msg = data.get("Information") or data.get("Note") or str(data)
        print(f"    No 'feed' key for {ticker}: {msg[:120]}")
        return []

    articles = []
    for item in data["feed"]:
        # Check that the target ticker is actually mentioned (not just incidental)
        ts_list = item.get("ticker_sentiment", [])
        relevance = 0.0
        for ts in ts_list:
            if ts.get("ticker", "").upper() == ticker.upper():
                try:
                    relevance = float(ts.get("relevance_score", 0))
                except ValueError:
                    pass
                break

        if relevance < MIN_RELEVANCE:
            continue

        # Convert time_published  "20240115T143000"  →  UNIX timestamp (seconds)
        time_str = item.get("time_published", "")
        try:
            dt = pd.to_datetime(time_str, format="%Y%m%dT%H%M%S", utc=True)
            unix_ts = int(dt.timestamp())
        except Exception:
            continue  # skip articles with unparseable dates

        articles.append({
            "ticker":    ticker,
            "title":     item.get("title",   ""),
            "summary":   item.get("summary", ""),
            "link":      item.get("url",     ""),
            "published": unix_ts,
        })

    return articles


# ── Main ──────────────────────────────────────────────────────────────────────
print(f"Alpha Vantage News download — {len(TICKERS)} tickers × {len(TIME_WINDOWS)} windows")
print(f"Output dir: {os.path.abspath(output_dir)}\n")

total_saved = 0

for ticker in TICKERS:
    all_articles = []

    for time_from, time_to in TIME_WINDOWS:
        print(f"  {ticker}  [{time_from} → {time_to}]", end=" ... ", flush=True)
        batch = fetch_news_window(ticker, time_from, time_to)
        print(f"{len(batch)} articles")
        all_articles.extend(batch)

        # Be polite to the API — 1 second between calls
        time.sleep(1)

    # Deduplicate by (title, published) in case windows overlap
    df = pd.DataFrame(all_articles)
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "published"])
        df = df.sort_values("published").reset_index(drop=True)

    save_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(save_path, index=False)
    total_saved += len(df)
    print(f"  → Saved {len(df)} rows to {save_path}\n")

print("=" * 55)
print(f"Total articles saved: {total_saved}")
print(f"Average per ticker  : {total_saved / len(TICKERS):.0f}")