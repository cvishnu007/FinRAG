"""
download_news_retry.py
======================
Re-downloads news for tickers whose CSV was empty after the first run.
Uses a 3-second delay between calls to avoid rate limiting.
"""

import requests, pandas as pd, os, time

API_KEY = "UHB2STC11ULRYEHB"

# Only the tickers that came back empty
EMPTY_TICKERS = ["AVGO", "BAC", "COST", "DIS", "KO", "MA", "PFE", "XOM"]

TIME_WINDOWS = [
    ("20190101T0000", "20220101T0000"),
    ("20220101T0000", "20260314T0000"),
]
MIN_RELEVANCE = 0.1
BASE_URL = "https://www.alphavantage.co/query"

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "data", "raw", "news")

def fetch_news_window(ticker, time_from, time_to):
    params = {
        "function": "NEWS_SENTIMENT", "tickers": ticker,
        "time_from": time_from, "time_to": time_to,
        "sort": "EARLIEST", "limit": 1000, "apikey": API_KEY,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        data = resp.json()
    except Exception as e:
        print(f"    Error: {e}")
        return []

    if "feed" not in data:
        msg = data.get("Information") or data.get("Note") or str(data)
        print(f"    No feed: {msg[:100]}")
        return []

    articles = []
    for item in data["feed"]:
        relevance = 0.0
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker", "").upper() == ticker.upper():
                try: relevance = float(ts.get("relevance_score", 0))
                except: pass
                break
        if relevance < MIN_RELEVANCE:
            continue
        try:
            dt = pd.to_datetime(item["time_published"], format="%Y%m%dT%H%M%S", utc=True)
            unix_ts = int(dt.timestamp())
        except:
            continue
        articles.append({
            "ticker": ticker, "title": item.get("title", ""),
            "summary": item.get("summary", ""), "link": item.get("url", ""),
            "published": unix_ts,
        })
    return articles

print(f"Retrying {len(EMPTY_TICKERS)} empty tickers with 3s delay between calls...\n")

total = 0
for ticker in EMPTY_TICKERS:
    all_articles = []
    for time_from, time_to in TIME_WINDOWS:
        print(f"  {ticker}  [{time_from} → {time_to}]", end=" ... ", flush=True)
        batch = fetch_news_window(ticker, time_from, time_to)
        print(f"{len(batch)} articles")
        all_articles.extend(batch)
        time.sleep(3)   # longer delay to stay under rate limit

    df = pd.DataFrame(all_articles)
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "published"])
        df = df.sort_values("published").reset_index(drop=True)

    save_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(save_path, index=False)
    total += len(df)
    print(f"  → Saved {len(df)} rows\n")

print(f"Done. Total additional articles: {total}")
