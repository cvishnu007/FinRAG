import yfinance as yf
import os

tickers = [
    "AAPL","MSFT","AMZN","GOOGL","TSLA",
    "NVDA","META","JPM","V","UNH",
    "HD","PG","MA","BAC","XOM",
    "AVGO","PFE","KO","COST","DIS"
]

start_date = "2019-01-01"
end_date = "2026-03-14"  # Updated to cover current news timestamps (Feb 2026)

# Use script-relative absolute path so script works from any working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "data", "raw", "prices")
os.makedirs(output_dir, exist_ok=True)

for ticker in tickers:
    print("Downloading:", ticker)
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)

    # Flatten MultiIndex columns that yfinance sometimes produces
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    save_path = os.path.join(output_dir, f"{ticker}.csv")
    df.to_csv(save_path, index=False)
    print(f"  Saved {len(df)} rows to {save_path}")
