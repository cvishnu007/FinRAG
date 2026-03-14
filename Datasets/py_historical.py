import requests
import pandas as pd
from tqdm import tqdm

def scrape_yahoo_news(ticker, pages=5):
    articles = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for page in tqdm(range(pages)):
        offset = page * 10

        url = (
            f"https://query2.finance.yahoo.com/v1/finance/search?"
            f"q={ticker}&newsCount=10&offset={offset}"
        )

        response = requests.get(url, headers=headers)
        data = response.json()

        if "news" not in data:
            continue

        for item in data["news"]:
            articles.append({
                "ticker": ticker,
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "link": item.get("link"),
                "published": item.get("providerPublishTime")
            })

    return pd.DataFrame(articles)


df = scrape_yahoo_news("AAPL", pages=20)
df.to_csv("aapl_news.csv", index=False)

print("Collected:", len(df), "articles")
