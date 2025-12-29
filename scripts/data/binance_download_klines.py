#!/usr/bin/env python3
"""
Download raw Binance klines (public, no API keys) and save to CSV.GZ.

Example:
  python scripts/binance_download_klines.py \
    --symbol BTCUSDT --interval 1h \
    --start 2024-06-01 --end 2024-09-30 \
    --out data/BTCUSDT_1H_real.csv.gz

Notes:
- Uses Binance /api/v3/klines (max 1000 candles per request)
- Public endpoint: no keys required
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple

import requests


BINANCE_BASE = "https://api.binance.com"


def parse_date_utc(s: str) -> int:
    """
    Parse YYYY-MM-DD or ISO datetime into UTC milliseconds.
    """
    s = s.strip()
    # Allow YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    # Allow ISO-like: 2024-06-01T00:00:00Z or without Z
    if s.endswith("Z"):
        s = s[:-1]
    # Try parsing with seconds
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(f"Unsupported date format: {s}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def interval_to_ms(interval: str) -> int:
    """
    Binance interval strings: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    unit = interval[-1]
    n = int(interval[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    if unit == "w":
        return n * 7 * 86_400_000
    if unit == "M":
        # Approx. month for stepping; Binance handles calendar months internally.
        return n * 30 * 86_400_000
    raise ValueError(f"Unsupported interval: {interval}")


@dataclass
class Kline:
    timestamp: int  # open time (ms)
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    timeout_s: int = 30,
    max_retries: int = 6,
) -> List[Kline]:
    """
    Fetch up to `limit` klines starting at start_ms (inclusive), stopping before end_ms (exclusive).
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=timeout_s)
            if r.status_code == 429:
                # rate limit
                retry_after = float(r.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, backoff))
                backoff = min(backoff * 1.8, 20.0)
                continue
            r.raise_for_status()
            data = r.json()
            out: List[Kline] = []
            for row in data:
                # row format:
                # [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, numberOfTrades,
                #   takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, ignore ]
                out.append(
                    Kline(
                        timestamp=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                )
            return out
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 20.0)

    return []  # unreachable


def iter_all_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    throttle_s: float = 0.2,
) -> Iterable[Kline]:
    """
    Iterate through all klines between [start_ms, end_ms).
    """
    step_ms = interval_to_ms(interval)
    session = requests.Session()

    t = start_ms
    last_ts: Optional[int] = None

    while t < end_ms:
        batch = fetch_klines(session, symbol, interval, t, end_ms, limit=1000)
        if not batch:
            # No data returned; step forward by one interval to avoid infinite loops.
            t += step_ms
            time.sleep(throttle_s)
            continue

        for k in batch:
            # Guard against duplicates / non-monotonic returns
            if last_ts is not None and k.timestamp <= last_ts:
                continue
            last_ts = k.timestamp
            yield k

        # Next start time = last timestamp + interval
        t = batch[-1].timestamp + step_ms
        time.sleep(throttle_s)


def write_csv_gz(path: str, rows: Iterable[Kline]) -> int:
    """
    Write rows to a gzipped CSV with columns:
    timestamp,open,high,low,close,volume
    """
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for k in rows:
            w.writerow([k.timestamp, f"{k.open:.8f}", f"{k.high:.8f}", f"{k.low:.8f}", f"{k.close:.8f}", f"{k.volume:.8f}"])
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ap.add_argument("--interval", required=True, help="e.g. 1h, 15m, 1d")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--out", required=True, help="Output path, e.g. data/BTCUSDT_1H_real.csv.gz")
    ap.add_argument("--throttle", type=float, default=0.2, help="Sleep between requests (seconds)")
    args = ap.parse_args()

    start_ms = parse_date_utc(args.start)
    end_ms = parse_date_utc(args.end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    rows = iter_all_klines(args.symbol, args.interval, start_ms, end_ms, throttle_s=args.throttle)
    n = write_csv_gz(args.out, rows)
    print(f"Wrote {n} klines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

