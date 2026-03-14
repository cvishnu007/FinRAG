"""
download_news_yahoo_extended.py
================================
Downloads news for the 8 tickers that had no Alpha Vantage data,
using two complementary Yahoo Finance approaches:

  1. yfinance Ticker.news  -- recent articles (no key required)
  2. Yahoo Finance search API with multiple keyword queries -- more articles

Both results are merged and deduplicated before saving.
"""

import requests
import pandas as pd
import os
import time
import yfinance as yf

TICKERS = ["AVGO", "BAC", "COST", "DIS", "KO", "MA", "PFE", "XOM"]

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "data", "raw", "news")
os.makedirs(output_dir, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# Multiple search queries per ticker to get more diverse articles
EXTRA_QUERIES = {
    "AVGO": ["Broadcom", "AVGO semiconductor", "Broadcom AI chips"],
    "BAC":  ["Bank of America", "BAC stock", "Bank of America earnings"],
    "COST": ["Costco", "COST warehouse", "Costco earnings"],
    "DIS":  ["Disney", "DIS stock", "Disney streaming"],
    "KO":   ["Coca Cola", "KO stock", "Coca Cola earnings"],
    "MA":   ["Mastercard", "MA stock", "Mastercard payments"],
    "PFE":  ["Pfizer", "PFE stock", "Pfizer drug"],
    "XOM":  ["ExxonMobil", "XOM oil", "Exxon earnings"],
}


def fetch_yfinance_news(ticker: str) -> list[dict]:
    """Use yfinance Ticker.news to get recent articles."""
    articles = []
    try:
        t = yf.Ticker(ticker)
        news_items = t.news  # returns list of dicts
        for item in (news_items or []):
            pub = item.get("providerPublishTime") or item.get("published")
            if not pub:
                continue
            articles.append({
                "ticker":    ticker,
                "title":     item.get("title", ""),
                "summary":   item.get("summary", ""),
                "link":      item.get("link", ""),
                "published": int(pub),
            })
    except Exception as e:
        print(f"    yfinance error for {ticker}: {e}")
    return articles


def fetch_yahoo_search(query: str, ticker: str, pages: int = 20) -> list[dict]:
    """Scrape Yahoo Finance search API with a keyword query."""
    articles = []
    for page in range(pages):
        offset = page * 10
        url = (
            f"https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={requests.utils.quote(query)}&newsCount=10&start={offset}"
        )
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break
            data = resp.json()
            items = data.get("news", [])
            if not items:
                break
            for item in items:
                pub = item.get("providerPublishTime")
                if not pub:
                    continue
                articles.append({
                    "ticker":    ticker,
                    "title":     item.get("title", ""),
                    "summary":   item.get("summary", ""),
                    "link":      item.get("link", ""),
                    "published": int(pub),
                })
            time.sleep(0.3)
        except Exception as e:
            print(f"    Yahoo search error [{query}] page {page}: {e}")
            break
    return articles


print(f"Downloading news for {len(TICKERS)} tickers via Yahoo Finance...\n")

total_saved = 0

for ticker in TICKERS:
    print(f"Processing: {ticker}")
    all_articles = []

    # Source 1: yfinance
    batch = fetch_yfinance_news(ticker)
    print(f"  yfinance.news        : {len(batch)} articles")
    all_articles.extend(batch)
    time.sleep(0.5)

    # Source 2: Yahoo Finance search with multiple queries
    for query in EXTRA_QUERIES.get(ticker, [ticker]):
        batch = fetch_yahoo_search(query, ticker, pages=15)
        print(f"  Yahoo search [{query:<25}]: {len(batch)} articles")
        all_articles.extend(batch)
        time.sleep(0.5)

    # Deduplicate and save
    df = pd.DataFrame(all_articles)
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "published"])
        df = df.sort_values("published").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["ticker","title","summary","link","published"])

    save_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(save_path, index=False)
    total_saved += len(df)
    print(f"  => Saved {len(df)} rows to {save_path}\n")

print("=" * 50)
print(f"Total articles saved : {total_saved}")
print(f"Average per ticker   : {total_saved / len(TICKERS):.0f}")
