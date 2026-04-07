# save as scripts/diagnose.py and run: python scripts/diagnose.py

import pandas as pd
import numpy as np
import os

DATA_PATH = "data/processed/master_dataset.csv"
PRICES_DIR = "data/raw/prices"

print("=" * 60)
print("DIAGNOSTIC REPORT")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df["published"] = pd.to_datetime(df["published"], errors="coerce")
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
df = df.sort_values("published").reset_index(drop=True)
n = len(df)

# ── 1. Basic shape ────────────────────────────────────────────────────────────
print("\n── 1. DATASET SHAPE ──")
print(f"  Total rows       : {n:,}")
print(f"  Columns          : {df.columns.tolist()}")
print(f"  Date range       : {df['published'].min()} → {df['published'].max()}")
print(f"  Tickers          : {sorted(df['ticker'].unique().tolist())}")
print(f"  Label balance    : UP={df['label'].mean()*100:.1f}%  DOWN={(1-df['label'].mean())*100:.1f}%")

# ── 2. Split boundaries ───────────────────────────────────────────────────────
print("\n── 2. SPLIT BOUNDARIES ──")
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
train_end   = df.iloc[n_train - 1]["published"]
val_start   = df.iloc[n_train]["published"]
val_end     = df.iloc[n_train + n_val - 1]["published"]
test_start  = df.iloc[n_train + n_val]["published"]
test_end    = df.iloc[-1]["published"]
gap_tv = (val_start - train_end).days
gap_vt = (test_start - val_end).days

print(f"  Train : {df.iloc[0]['published'].date()} → {train_end.date()}  ({n_train:,} rows)")
print(f"  Val   : {val_start.date()} → {val_end.date()}  ({n_val:,} rows)   [gap from train: {gap_tv} days]")
print(f"  Test  : {test_start.date()} → {test_end.date()}  ({n - n_train - n_val:,} rows)   [gap from val: {gap_vt} days]")
print(f"  ⚠  Gap warnings: {'YES — gaps < 7 days, soft leakage risk' if gap_tv < 7 or gap_vt < 7 else 'OK'}")

# ── 3. Per-ticker counts in each split ───────────────────────────────────────
print("\n── 3. PER-TICKER ROW COUNTS PER SPLIT ──")
train_df = df.iloc[:n_train]
val_df   = df.iloc[n_train:n_train + n_val]
test_df  = df.iloc[n_train + n_val:]

counts = pd.DataFrame({
    "train" : train_df.groupby("ticker").size(),
    "val"   : val_df.groupby("ticker").size(),
    "test"  : test_df.groupby("ticker").size(),
}).fillna(0).astype(int)
counts["test_pct"] = (counts["test"] / counts["test"].sum() * 100).round(1)
print(counts.to_string())
thin_tickers = counts[counts["test"] < 50].index.tolist()
if thin_tickers:
    print(f"  ⚠  Tickers with < 50 test rows (unreliable metrics): {thin_tickers}")

# ── 4. News timestamp staleness ───────────────────────────────────────────────
print("\n── 4. NEWS TIMESTAMP STALENESS ──")
df["lag_days"] = (df["trade_date"] - df["published"].dt.normalize()).dt.days
lag_dist = df["lag_days"].value_counts().sort_index()
print("  lag_days distribution (news → trade_date):")
print("  " + lag_dist.head(15).to_string().replace("\n", "\n  "))
stale = (df["lag_days"] >= 3).mean() * 100
print(f"\n  % articles with lag ≥ 3 days : {stale:.1f}%")
if stale > 20:
    print("  ⚠  HIGH staleness — many articles arrive after market has already moved!")
else:
    print("  ✓  Staleness looks OK")

# ── 5. Label leakage check ────────────────────────────────────────────────────
print("\n── 5. LABEL LEAKAGE CHECK ──")
dangerous_cols = [c for c in df.columns if c in {"return_1d", "Close", "trade_date"}]
print(f"  Columns that must NOT enter the model: {dangerous_cols}")
print("  (Verify these are absent from cat_cols/num_cols/text_col in train_baseline.py)")

# ── 6. Text quality ───────────────────────────────────────────────────────────
print("\n── 6. TEXT QUALITY ──")
df["text"] = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.strip()
df["text_len"] = df["text"].str.len()
empty_text = (df["text_len"] == 0).sum()
short_text  = (df["text_len"] < 30).sum()
print(f"  Empty text rows  : {empty_text}")
print(f"  Very short (<30 chars) : {short_text}")
print(f"  Median text length     : {df['text_len'].median():.0f} chars")
print(f"  Mean text length       : {df['text_len'].mean():.0f} chars")

# ── 7. Duplicate detection ────────────────────────────────────────────────────
print("\n── 7. DUPLICATE DETECTION ──")
dup_full = df.duplicated(subset=["ticker", "title", "published"]).sum()
dup_title = df.duplicated(subset=["title"]).sum()
print(f"  Exact duplicates (ticker+title+published) : {dup_full}")
print(f"  Same title across tickers                 : {dup_title}")
if dup_title > 0:
    print("  ⚠  Cross-ticker duplicates detected — same news event counted multiple times")

# ── 8. Market feature sanity ──────────────────────────────────────────────────
print("\n── 8. MARKET FEATURE SANITY ──")
if "ret_1d" in df.columns:
    print(f"  ret_1d present: YES")
    inf_count = np.isinf(df["ret_1d"]).sum()
    nan_count  = df["ret_1d"].isna().sum()
    print(f"    NaN  : {nan_count}")
    print(f"    Inf  : {inf_count}")
    print(f"    Mean : {df['ret_1d'].mean():.4f}")
    print(f"    Std  : {df['ret_1d'].std():.4f}")
else:
    print("  ret_1d not in dataset (added at training time — OK)")

# ── 9. Price file coverage ────────────────────────────────────────────────────
print("\n── 9. PRICE FILE COVERAGE ──")
for ticker in sorted(df["ticker"].unique()):
    path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        pdf = pd.read_csv(path)
        print(f"  {ticker:6s} : {len(pdf):,} rows  ✓")
    else:
        print(f"  {ticker:6s} : MISSING ✗")

# ── 10. Return distribution by ticker ────────────────────────────────────────
print("\n── 10. LABEL DISTRIBUTION PER TICKER ──")
label_by_ticker = df.groupby("ticker")["label"].agg(["mean", "count"])
label_by_ticker.columns = ["up_pct", "count"]
label_by_ticker["up_pct"] = (label_by_ticker["up_pct"] * 100).round(1)
print(label_by_ticker.to_string())
skewed = label_by_ticker[abs(label_by_ticker["up_pct"] - 50) > 10].index.tolist()
if skewed:
    print(f"  ⚠  Tickers with skewed labels (>10% off 50/50): {skewed}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE — paste output above for analysis")
print("=" * 60)