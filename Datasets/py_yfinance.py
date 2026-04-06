import yfinance as yf
import pprint

ticker = yf.Ticker("AAPL")
news = ticker.news

#pprint.pprint(news[0])

import pandas as pd

rows = []

for article in news:
    content = article.get("content", {})
    rows.append({
        "ticker": "AAPL",
        "title": content.get("title"),
        "summary": content.get("summary"),
        "date": content.get("pubDate"),
        "publisher": content.get("provider", {}).get("displayName"),
        "url": content.get("canonicalUrl", {}).get("url")
    })

df = pd.DataFrame(rows)
df.to_csv("aapl2_news.csv", index=False)
