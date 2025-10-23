#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_prices_csv.py
Stáhne OHLCV z Binance a uloží CSV s kolonami: time, close.
"""
import argparse, sys, requests
import pandas as pd

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol: str, interval: str, limit: int):
    params = dict(symbol=symbol.upper(), interval=interval, limit=limit)
    r = requests.get(BINANCE_KLINES, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    return df[["open_time","close"]].rename(columns={"open_time":"time"})

def parse_args():
    p = argparse.ArgumentParser(description="Uložení Binance cen do CSV (time,close)")
    p.add_argument("--symbols", required=True, type=str, help="např. BTCUSDT,ETHUSDT")
    p.add_argument("--interval", default="1h", type=str)
    p.add_argument("--limit", default=3000, type=int)
    p.add_argument("--outdir", default="prices", type=str)
    return p.parse_args()

def main():
    args = parse_args()
    import os
    os.makedirs(args.outdir, exist_ok=True)
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        df = fetch_klines(sym, args.interval, args.limit)
        path = f"{args.outdir}/{sym}_{args.interval}.csv"
        df.to_csv(path, index=False)
        print(f"Uloženo: {path}  ({len(df)} řádků)")

if __name__ == "__main__":
    main()
