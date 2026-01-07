#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

import ccxt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--symbols", default="BTC/USDT,ETH/USDT,XRP/USDT,DOT/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--out-dir", default="data/ohlcv_1h")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    exchange = getattr(ccxt, args.exchange)({
        "enableRateLimit": True,
    })

    since_ms = exchange.parse8601(args.since + "T00:00:00Z")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sym in args.symbols.split(","):
        sym = sym.strip()
        print(f"Fetching {sym} {args.timeframe} ...")

        all_rows = []
        since = since_ms

        while True:
            ohlcv = exchange.fetch_ohlcv(
                sym,
                timeframe=args.timeframe,
                since=since,
                limit=args.limit,
            )
            if not ohlcv:
                break

            all_rows.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)

            if len(ohlcv) < args.limit:
                break

        df = pd.DataFrame(
            all_rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp")

        fname = sym.replace("/", "") + ".csv"
        path = out_dir / fname
        df.to_csv(path, index=False)
        print(f"Saved {path} ({len(df)} bars)")

    print("Done.")


if __name__ == "__main__":
    main()

