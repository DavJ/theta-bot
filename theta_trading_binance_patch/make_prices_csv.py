#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_prices_csv.py
Stáhne historické OHLCV z Binance přes ccxt a uloží CSV:
prices/<SYMBOL>_<INTERVAL>.csv
"""
import argparse, os
from pathlib import Path
import ccxt
import pandas as pd

INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w",
}

def fetch_ohlcv_binance(symbol: str, timeframe: str, limit: int):
    ex = ccxt.binance({"enableRateLimit": True})
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not data:
        raise RuntimeError(f"No data for {symbol} {timeframe}")
    df = pd.DataFrame(data, columns=["t","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df[["time","open","high","low","close","volume"]].copy()
    df = df.dropna().reset_index(drop=True)
    df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df

def norm_symbol_for_ccxt(sym: str) -> str:
    # BTCUSDT -> BTC/USDT (heuristika pro XXXUSDT/USDC/FDUSD apod.)
    if "/" in sym:
        return sym
    bases = ["USDT","USDC","FDUSD","BUSD","TUSD"]
    for q in bases:
        if sym.endswith(q):
            base = sym[:-len(q)]
            return f"{base}/{q}"
    # fallback: poslední 4 znaky jako quote
    return f"{sym[:-4]}/{sym[-4:]}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="CSV seznam tickerů, např. BTCUSDT,ETHUSDT")
    ap.add_argument("--interval", required=True, choices=INTERVAL_MAP.keys())
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--outdir", default="prices")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    tf = INTERVAL_MAP[args.interval]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    for sym in symbols:
        sym_ccxt = norm_symbol_for_ccxt(sym)
        df = fetch_ohlcv_binance(sym_ccxt, tf, args.limit)
        out = Path(args.outdir) / f"{sym}_{args.interval}.csv"
        df.to_csv(out, index=False)
        print(f"Uloženo: {out}  ({len(df)} řádků)")

if __name__ == "__main__":
    # závislost: pip install ccxt pandas
    main()
