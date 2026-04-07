from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def _parse_time_hhmm(value: str) -> pd.Timedelta:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid HH:MM time: {value}")
    hours = int(parts[0])
    minutes = int(parts[1])
    return pd.Timedelta(hours=hours, minutes=minutes)


def _load_price_features(path: str, market_close: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"Missing Date/Close columns in {path}")

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

    close_time = _parse_time_hhmm(market_close)
    df["close_ts"] = df["Date"] + close_time

    daily_ret = df["Close"].pct_change()
    df["ret_1d"] = daily_ret.shift(1)
    df["ret_3d"] = daily_ret.rolling(3).sum().shift(1)
    df["ret_5d"] = daily_ret.rolling(5).sum().shift(1)
    df["roll_mean_5"] = daily_ret.rolling(5).mean().shift(1)
    df["roll_vol_5"] = daily_ret.rolling(5).std().shift(1)

    return df[[
        "close_ts",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "roll_mean_5",
        "roll_vol_5",
    ]]


def add_market_features(
    df: pd.DataFrame,
    prices_dir: str,
    timezone: str,
    market_close: str,
) -> pd.DataFrame:
    if not os.path.isdir(prices_dir):
        raise FileNotFoundError(f"prices_dir not found: {prices_dir}")

    df = df.copy()
    published = pd.to_datetime(df["published"], errors="coerce")
    if published.dt.tz is None:
        published = published.dt.tz_localize("UTC")
    published = published.dt.tz_convert(timezone)
    df["published_tz"] = published

    out_frames = []

    for ticker, group in df.groupby("ticker", sort=False):
        price_path = os.path.join(prices_dir, f"{ticker}.csv")
        if not os.path.exists(price_path):
            group = group.copy()
            group[["ret_1d", "ret_3d", "ret_5d", "roll_mean_5", "roll_vol_5"]] = 0.0
            out_frames.append(group)
            continue

        price_df = _load_price_features(price_path, market_close)
        price_df["close_ts"] = price_df["close_ts"].dt.tz_localize(timezone)

        group = group.sort_values("published_tz")
        merged = pd.merge_asof(
            group,
            price_df,
            left_on="published_tz",
            right_on="close_ts",
            direction="backward",
        )
        merged = merged.drop(columns=["close_ts"])
        out_frames.append(merged)

    out = pd.concat(out_frames, axis=0).sort_index()
    return out
