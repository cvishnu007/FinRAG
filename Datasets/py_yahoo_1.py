import feedparser
import pandas as pd
from tqdm import tqdm

tickers = ["AAPL", "MSFT", "TSLA"]

all_news = []

for ticker in tqdm(tickers):
    rss_url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    feed = feedparser.parse(rss_url)

    for entry in feed.entries:
        all_news.append({
            "ticker": ticker,
            "title": entry.title,
            "summary": entry.summary,
            "published": entry.published,
            "link": entry.link
        })

df = pd.DataFrame(all_news)
df.to_csv("yahoo_news_raw.csv", index=False)

print("Saved:", len(df), "articles")
