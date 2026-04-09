import yfinance as yf
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "..", "data", "raw", "prices")
os.makedirs(output_dir, exist_ok=True)

ticker = "SPY"
print("Downloading:", ticker)
df = yf.download(ticker, start="2019-01-01", end="2026-03-14", auto_adjust=True)

if hasattr(df.columns, "levels"):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()
save_path = os.path.join(output_dir, f"{ticker}.csv")
df.to_csv(save_path, index=False)
print(f"Saved {len(df)} rows to {save_path}")