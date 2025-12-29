#!/usr/bin/env python3
"""
Download Binance futures open interest history and save to CSV.GZ.

Open interest represents the total number of outstanding derivative contracts.

Example:
  python scripts/data/binance_download_open_interest_hist.py \
    --symbol BTCUSDT \
    --period 1h \
    --start 2024-01-01 --end 2024-10-01 \
    --out data/raw/futures/BTCUSDT_oi.csv.gz

Endpoint: /futures/data/openInterestHist
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


def period_to_ms(period: str) -> int:
    """
    Convert period string to milliseconds.
    Supported: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
    """
    unit = period[-1]
    n = int(period[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 3_600_000
    if unit == "d":
        return n * 86_400_000
    raise ValueError(f"Unsupported period: {period}")


@dataclass
class OpenInterest:
    timestamp: int  # time (ms)
    sum_open_interest: float  # Total open interest in base asset
    sum_open_interest_value: float  # Total open interest in USDT
    symbol: str


def fetch_open_interest_hist(
    session: requests.Session,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    limit: int = 500,
    timeout_s: int = 30,
    max_retries: int = 6,
) -> List[OpenInterest]:
    """
    Fetch open interest history from Binance futures API.
    
    Note: This endpoint has a max limit of 500 records per request.
    """
    url = f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist"
    params = {
        "symbol": symbol.upper(),
        "period": period,
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
            
            out: List[OpenInterest] = []
            for row in data:
                # Response format: {"symbol": "BTCUSDT", "sumOpenInterest": "123456.78", 
                #                   "sumOpenInterestValue": "5200000000.00", "timestamp": 1640995200000}
                out.append(
                    OpenInterest(
                        timestamp=int(row["timestamp"]),
                        sum_open_interest=float(row["sumOpenInterest"]),
                        sum_open_interest_value=float(row["sumOpenInterestValue"]),
                        symbol=row["symbol"],
                    )
                )
            return out
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 20.0)

    return []


def iter_all_open_interest(
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    throttle_s: float = 0.2,
) -> Iterable[OpenInterest]:
    """
    Iterate through all open interest records between [start_ms, end_ms).
    """
    period_ms = period_to_ms(period)
    session = requests.Session()
    
    t = start_ms
    last_ts: Optional[int] = None

    while t < end_ms:
        batch = fetch_open_interest_hist(session, symbol, period, t, end_ms, limit=500)
        if not batch:
            # No data; step forward
            t += period_ms
            time.sleep(throttle_s)
            continue

        for oi in batch:
            if last_ts is not None and oi.timestamp <= last_ts:
                continue
            last_ts = oi.timestamp
            yield oi

        # Next start time
        t = batch[-1].timestamp + 1
        time.sleep(throttle_s)


def write_csv_gz(path: str, rows: Iterable[OpenInterest]) -> int:
    """
    Write open interest records to gzipped CSV with columns:
    timestamp,sumOpenInterest,sumOpenInterestValue,symbol
    """
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "sumOpenInterest", "sumOpenInterestValue", "symbol"])
        for oi in rows:
            w.writerow([
                oi.timestamp, 
                f"{oi.sum_open_interest:.8f}", 
                f"{oi.sum_open_interest_value:.8f}", 
                oi.symbol
            ])
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Binance futures open interest history"
    )
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ap.add_argument("--period", required=True, help="e.g. 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--out", required=True, help="Output path, e.g. data/raw/futures/BTCUSDT_oi.csv.gz")
    ap.add_argument("--throttle", type=float, default=0.2, help="Sleep between requests (seconds)")
    args = ap.parse_args()

    start_ms = parse_date_utc(args.start)
    end_ms = parse_date_utc(args.end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    print(f"Downloading open interest history for {args.symbol} (period={args.period}) from {args.start} to {args.end}")
    rows = iter_all_open_interest(args.symbol, args.period, start_ms, end_ms, throttle_s=args.throttle)
    n = write_csv_gz(args.out, rows)
    print(f"Wrote {n} open interest records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
