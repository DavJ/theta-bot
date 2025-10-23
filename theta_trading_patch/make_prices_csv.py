#!/usr/bin/env python3
import argparse, time
from pathlib import Path
import pandas as pd

try:
    import ccxt
except ImportError as e:
    raise SystemExit("Chybí ccxt. Nainstaluj: pip install ccxt") from e

BINANCE = ccxt.binance({
    "enableRateLimit": True,
})

def fetch_ohlcv(symbol: str, interval: str, limit: int):
    data = BINANCE.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
    return data

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df[["time", "open", "high", "low", "close", "volume"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated tickery, např. BTCUSDT,ETHUSDT")
    ap.add_argument("--interval", required=True, help="Binance/ccxt timeframe, např. 1h, 15m, 1d")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--outdir", default="prices")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        print(f"Stahuji {sym} {args.interval} (limit={args.limit}) …")
        ohlcv = fetch_ohlcv(sym, args.interval, args.limit)
        df = to_df(ohlcv)
        out = outdir / f"{sym}_{args.interval}.csv"
        df.to_csv(out, index=False)
        print(f"Uloženo: {out}  ({len(df)} řádků)")
        time.sleep(getattr(BINANCE, "rateLimit", 300)/1000)

if __name__ == "__main__":
    main()
