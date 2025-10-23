#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Download OHLCV closes from Binance via ccxt and save CSVs to an output folder.
# Usage:
#   python make_prices_csv.py --symbols BTCUSDT,ETHUSDT --interval 1h --limit 1000 --outdir prices
# Output CSV columns: time, close
# File name: <outdir>/<SYMBOL>_<interval>.csv

import argparse
import os
from pathlib import Path
from datetime import datetime, timezone
import sys

try:
    import ccxt  # type: ignore
    import pandas as pd  # noqa: F401 (used when saving)
except Exception as e:
    print("[error] Missing dependency:", e, file=sys.stderr)
    print("Try: pip install ccxt pandas", file=sys.stderr)
    sys.exit(1)

TIMEFRAMES = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
    "1d":"1d","3d":"3d","1w":"1w","1M":"1M"
}

# naive parser to infer base/quote from a common concatenated symbol (e.g., BTCUSDT)
COMMON_QUOTES = ("USDT","USDC","BUSD","FDUSD","TUSD","BTC","ETH","BNB")

def normalize_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    s = sym.upper().strip()
    for q in COMMON_QUOTES:
        if s.endswith(q):
            base = s[:-len(q)]
            if base and base.isalpha():
                return f"{base}/{q}"
    # fallback: insert slash before last 4 chars
    if len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s

def iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--interval", required=True, help="Binance timeframe like 1h, 15m, 1d")
    ap.add_argument("--limit", type=int, default=1000, help="Number of candles to fetch (per symbol)")
    ap.add_argument("--outdir", default="prices", help="Output directory for CSV files (default: prices)")
    args = ap.parse_args()

    tf = args.interval
    if tf not in TIMEFRAMES:
        print(f"[error] Unsupported interval '{tf}'. Supported: {sorted(TIMEFRAMES.keys())}", file=sys.stderr)
        sys.exit(2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exch = ccxt.binance({"enableRateLimit": True})
    exch.load_markets()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        mkt = normalize_symbol(sym)
        try:
            ohlcv = exch.fetch_ohlcv(mkt, timeframe=tf, limit=args.limit)
        except Exception as e:
            print(f"[warn] Failed to fetch {sym} as {mkt}: {e}", file=sys.stderr)
            continue
        if not ohlcv:
            print(f"[warn] Empty data for {sym}", file=sys.stderr)
            continue
        # columns: [timestamp, open, high, low, close, volume]
        times = [iso(r[0]) for r in ohlcv]
        closes = [r[4] for r in ohlcv]
        import pandas as pd
        df = pd.DataFrame({"time": times, "close": closes})
        out_path = outdir / f"{sym.upper()}_{tf}.csv"
        df.to_csv(out_path, index=False)
        print(f"Uloženo: {out_path}  ({len(df)} řádků)")

if __name__ == "__main__":
    main()
