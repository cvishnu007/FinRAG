import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Use script-relative absolute paths so the script works regardless of the
# working directory from which it is invoked.
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_dir  = os.path.join(script_dir, "..")

price_folder = os.path.join(project_dir, "data", "raw", "prices")
news_folder  = os.path.join(project_dir, "data", "raw", "news")
out_folder   = os.path.join(project_dir, "data", "processed")
os.makedirs(out_folder, exist_ok=True)

# ── Counters ──────────────────────────────────────────────────────────────────
merged_rows              = []
total_news_rows          = 0
total_matched            = 0
total_skipped_no_price   = 0   # price file missing for this ticker
total_skipped_no_future  = 0   # no trading day on/after the news timestamp
total_skipped_na_return  = 0   # next-day return is NaN (last row of price data)

# ── Helper: load a price CSV robustly ─────────────────────────────────────────
def load_price_csv(path):
    """
    yfinance CSVs sometimes contain a second 'header' row with the ticker name
    (a MultiIndex artefact).  We detect and drop that row so numerics parse
    correctly.
    """
    df = pd.read_csv(path, header=0)

    # Drop any row where 'Date' is literally the string 'Ticker' or 'Price'
    # (artefact from yfinance MultiIndex dumps)
    if df.shape[0] > 0 and not pd.to_numeric(df.iloc[0]["Close"], errors="coerce") == df.iloc[0]["Close"]:
        try:
            float(df.iloc[0]["Close"])
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

    # Flatten column names in case they were written with an extra level
    df.columns = [str(c).strip() for c in df.columns]

    # Keep only the columns we need; tolerate extras
    needed = {"Date", "Close"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns in {path}. Found: {df.columns.tolist()}")

    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ── Helper: parse published timestamp flexibly ────────────────────────────────
def parse_published(series):
    """
    Handles two formats:
      • UNIX integer (seconds since epoch) – from live Yahoo Finance scrape
      • ISO / human-readable datetime string – if already converted
    """
    # Try numeric first
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        dt = pd.to_datetime(numeric, unit="s", utc=True).dt.tz_localize(None)
    else:
        dt = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
    return dt


# ── Main loop ─────────────────────────────────────────────────────────────────
news_files = [f for f in os.listdir(news_folder) if f.endswith(".csv")]

for file in sorted(news_files):
    ticker      = file.replace(".csv", "")
    price_path  = os.path.join(price_folder, f"{ticker}.csv")
    news_path   = os.path.join(news_folder,  file)

    print(f"\nProcessing: {ticker}")

    # ── Load news (skip empty CSVs gracefully) ─────────────────────────────
    if os.path.getsize(news_path) < 10:
        print(f"  SKIP – news file is empty (0 articles): {news_path}")
        continue
    try:
        news_df = pd.read_csv(news_path)
    except Exception as e:
        print(f"  SKIP – could not read news CSV: {e}")
        continue
    if news_df.empty:
        print(f"  SKIP – news dataframe is empty after load.")
        continue
    total_news_rows += len(news_df)

    # ── Validate news columns ──────────────────────────────────────────────
    for col in ("published",):
        if col not in news_df.columns:
            print(f"  SKIP – news file missing column '{col}'")
            total_skipped_no_price += len(news_df)
            continue

    # ── Load prices ────────────────────────────────────────────────────────
    if not os.path.exists(price_path):
        print(f"  SKIP – price file not found: {price_path}")
        total_skipped_no_price += len(news_df)
        continue

    try:
        price_df = load_price_csv(price_path)
    except Exception as e:
        print(f"  SKIP – could not load price CSV: {e}")
        total_skipped_no_price += len(news_df)
        continue

    if price_df.empty:
        print(f"  SKIP – price dataframe is empty after cleaning")
        total_skipped_no_price += len(news_df)
        continue

    # ── Compute next-day (forward) return ──────────────────────────────────
    # pct_change gives today's return relative to yesterday.
    # shift(-1) aligns it so row[t] = (Close[t+1] - Close[t]) / Close[t]
    price_df["return_1d"] = price_df["Close"].pct_change().shift(-1)

    # Diagnostic ranges
    print(f"  price range : {price_df['Date'].min().date()} → {price_df['Date'].max().date()}  ({len(price_df)} rows)")

    # ── Parse news timestamps ──────────────────────────────────────────────
    news_df["published"] = parse_published(news_df["published"])
    news_df = news_df.dropna(subset=["published"])

    print(f"  news  range : {news_df['published'].min()} → {news_df['published'].max()}  ({len(news_df)} rows)")

    # ── Match each news article to the next available trading day ──────────
    trading_dates = price_df["Date"].values  # numpy array for fast searchsorted

    for _, news_row in news_df.iterrows():
        news_time = news_row["published"]

        # Find first price row where Date >= news_time
        future_prices = price_df[price_df["Date"] >= news_time]

        if future_prices.empty:
            total_skipped_no_future += 1
            continue

        next_day = future_prices.iloc[0]

        if pd.isna(next_day["return_1d"]):
            # This is the last row in the price file – no next day available
            total_skipped_na_return += 1
            continue

        merged_rows.append({
            "ticker"    : ticker,
            "title"     : news_row.get("title",   ""),
            "summary"   : news_row.get("summary", ""),
            "published" : news_time,
            "trade_date": next_day["Date"].date(),
            "Close"     : round(next_day["Close"],     4),
            "return_1d" : round(next_day["return_1d"], 6),
            "label"     : int(next_day["return_1d"] > 0),
        })
        total_matched += 1

# ── Build master DataFrame ────────────────────────────────────────────────────
if merged_rows:
    master = pd.DataFrame(merged_rows)

    # Deduplicate: same ticker + same news publish time might map to the same
    # trading day more than once if duplicate news entries exist.
    before_dedup = len(master)
    master = master.drop_duplicates(subset=["ticker", "published", "trade_date"])
    after_dedup  = len(master)
    if before_dedup != after_dedup:
        print(f"\nDeduplication removed {before_dedup - after_dedup} duplicate rows.")

    master = master.sort_values(["ticker", "published"]).reset_index(drop=True)
else:
    master = pd.DataFrame(columns=[
        "ticker", "title", "summary", "published", "trade_date",
        "Close", "return_1d", "label"
    ])

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(out_folder, "master_dataset.csv")
master.to_csv(out_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  Output saved to : {out_path}")
print(f"  Final dataset size         : {len(master):>7,}")
print(f"  Total news rows processed  : {total_news_rows:>7,}")
print(f"  Matched rows               : {total_matched:>7,}")
print(f"  Skipped (no price file)    : {total_skipped_no_price:>7,}")
print(f"  Skipped (no future price)  : {total_skipped_no_future:>7,}")
print(f"  Skipped (return NaN)       : {total_skipped_na_return:>7,}")
print("=" * 55)

if len(master) == 0:
    print("\n⚠  Dataset is empty!")
    print("   Most likely cause: news timestamps fall OUTSIDE the price date range.")
    print("   Fix: re-run download_prices.py so its end_date covers the news dates.")
