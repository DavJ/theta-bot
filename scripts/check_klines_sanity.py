#!/usr/bin/env python3
"""
Sanity check a Binance kline CSV.GZ file (raw-ishness, gaps, autocorr).

Example:
  python scripts/check_klines_sanity.py --path data/BTCUSDT_1H_real.csv.gz --interval 1h
"""

from __future__ import annotations

import argparse
import gzip
import math
from datetime import datetime, timezone

import pandas as pd


def ms_to_utc_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def interval_to_seconds(interval: str) -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    if unit == "m":
        return n * 60
    if unit == "h":
        return n * 3600
    if unit == "d":
        return n * 86400
    raise ValueError(f"Unsupported interval: {interval}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--interval", required=True, help="e.g. 1h")
    args = ap.parse_args()

    df = pd.read_csv(args.path, compression="gzip")
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].astype("int64")

    print(f"Rows: {len(df)}")
    print(f"Start: {ms_to_utc_str(int(ts.iloc[0]))}")
    print(f"End:   {ms_to_utc_str(int(ts.iloc[-1]))}")
    print(f"Close min/max: {df['close'].min():.2f} / {df['close'].max():.2f}")

    # monotonic
    mono = (ts.diff().dropna() > 0).all()
    print(f"Timestamp strictly increasing: {mono}")

    # gap check
    dt_expected_ms = interval_to_seconds(args.interval) * 1000
    diffs = ts.diff().dropna()
    gaps = diffs[diffs != dt_expected_ms]
    print(f"Expected step (ms): {dt_expected_ms}")
    print(f"Non-standard steps count: {len(gaps)}")
    if len(gaps) > 0:
        # show a few examples
        ex = gaps.head(5)
        print("Examples of non-standard steps (ms):", ex.tolist())

    # return autocorr
    close = df["close"].astype(float)
    rets = (close.pct_change()).dropna()
    if len(rets) > 5:
        ac1 = rets.autocorr(lag=1)
        ac2 = rets.autocorr(lag=2)
        print(f"Return autocorr lag1: {ac1:.4f}, lag2: {ac2:.4f}")
        if abs(ac1) > 0.2:
            print("WARNING: High lag-1 autocorrelation. Data may be processed/smoothed or resampled oddly.")
    else:
        print("Not enough data for autocorr check.")

    # NaNs
    nan_any = df[["open","high","low","close","volume"]].isna().any().any()
    print(f"Any NaNs in OHLCV: {nan_any}")

    # basic OHLC consistency
    bad = ((df["high"] < df["low"]) | (df["open"] < df["low"]) | (df["open"] > df["high"]) | (df["close"] < df["low"]) | (df["close"] > df["high"])).sum()
    print(f"OHLC consistency violations: {int(bad)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

