#!/usr/bin/env python3
"""
Download Binance futures mark price klines and save to CSV.GZ.

Mark price is used for funding rate calculation and liquidations.

Example:
  python scripts/data/binance_download_mark_klines.py \
    --symbol BTCUSDT \
    --interval 1h \
    --start 2024-01-01 --end 2024-10-01 \
    --out data/raw/futures/BTCUSDT_mark_1h.csv.gz

Endpoint: /fapi/v1/markPriceKlines
Public endpoint - no API key required.
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
from typing import Iterable, List, Optional

import requests


BINANCE_FAPI_BASE = "https://fapi.binance.com"


def parse_date_utc(s: str) -> int:
    """
    Parse YYYY-MM-DD or ISO datetime into UTC milliseconds.
    """
    s = s.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if s.endswith("Z"):
        s = s[:-1]
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
        return n * 30 * 86_400_000
    raise ValueError(f"Unsupported interval: {interval}")


@dataclass
class MarkKline:
    timestamp: int  # open time (ms)
    open: float
    high: float
    low: float
    close: float


def fetch_mark_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1500,
    timeout_s: int = 30,
    max_retries: int = 6,
) -> List[MarkKline]:
    """
    Fetch mark price klines from Binance futures API.
    """
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/markPriceKlines"
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
                retry_after = float(r.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, backoff))
                backoff = min(backoff * 1.8, 20.0)
                continue
            r.raise_for_status()
            data = r.json()
            
            out: List[MarkKline] = []
            for row in data:
                # Format: [openTime, open, high, low, close, ...]
                out.append(
                    MarkKline(
                        timestamp=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                    )
                )
            return out
        except (requests.RequestException, json.JSONDecodeError, (IndexError, KeyError)) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 20.0)

    return []


def iter_all_mark_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    throttle_s: float = 0.2,
) -> Iterable[MarkKline]:
    """
    Iterate through all mark klines between [start_ms, end_ms).
    """
    step_ms = interval_to_ms(interval)
    session = requests.Session()

    t = start_ms
    last_ts: Optional[int] = None

    while t < end_ms:
        batch = fetch_mark_klines(session, symbol, interval, t, end_ms, limit=1500)
        if not batch:
            # No data; step forward
            t += step_ms
            time.sleep(throttle_s)
            continue

        for k in batch:
            if last_ts is not None and k.timestamp <= last_ts:
                continue
            last_ts = k.timestamp
            yield k

        # Next start time
        t = batch[-1].timestamp + step_ms
        time.sleep(throttle_s)


def write_csv_gz(path: str, rows: Iterable[MarkKline]) -> int:
    """
    Write mark klines to gzipped CSV with columns:
    timestamp,open,high,low,close
    """
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close"])
        for k in rows:
            w.writerow([k.timestamp, f"{k.open:.8f}", f"{k.high:.8f}", f"{k.low:.8f}", f"{k.close:.8f}"])
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Binance futures mark price klines"
    )
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ap.add_argument("--interval", required=True, help="e.g. 1h, 15m, 1d")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--out", required=True, help="Output path, e.g. data/raw/futures/BTCUSDT_mark_1h.csv.gz")
    ap.add_argument("--throttle", type=float, default=0.2, help="Sleep between requests (seconds)")
    args = ap.parse_args()

    start_ms = parse_date_utc(args.start)
    end_ms = parse_date_utc(args.end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    print(f"Downloading mark price klines for {args.symbol} ({args.interval}) from {args.start} to {args.end}")
    rows = iter_all_mark_klines(args.symbol, args.interval, start_ms, end_ms, throttle_s=args.throttle)
    n = write_csv_gz(args.out, rows)
    print(f"Wrote {n} mark klines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
