import yfinance as yf
import os

tickers = [
    "AAPL","MSFT","AMZN","GOOGL","TSLA",
    "NVDA","META","JPM","V","UNH",
    "HD","PG","MA","BAC","XOM",
    "AVGO","PFE","KO","COST","DIS"
]

start_date = "2019-01-01"
end_date = "2024-12-31"

os.makedirs("../data/raw/prices", exist_ok=True)

for ticker in tickers:
    print("Downloading:", ticker)
    df = yf.download(ticker, start=start_date, end=end_date)
    df.to_csv(f"../data/raw/prices/{ticker}.csv")
