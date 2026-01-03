#!/usr/bin/env python3
"""
Download Binance futures funding rate history and save to CSV.GZ.

Funding rates are published every 8 hours at 00:00, 08:00, 16:00 UTC.

Example:
  python scripts/data/binance_download_funding.py \
    --symbol BTCUSDT \
    --start 2024-01-01 --end 2024-10-01 \
    --out data/raw/futures/BTCUSDT_funding.csv.gz

Endpoint: /fapi/v1/fundingRate
Public endpoint - no API key required for historical data.
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
from pathlib import Path
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


@dataclass
class FundingRate:
    timestamp: int  # funding time (ms)
    funding_rate: float
    symbol: str


def fetch_funding_rates(
    session: requests.Session,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    timeout_s: int = 30,
    max_retries: int = 6,
) -> List[FundingRate]:
    """
    Fetch funding rate history from Binance futures API.
    
    Returns up to `limit` records starting from start_ms.
    """
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate"
    params = {
        "symbol": symbol.upper(),
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=timeout_s)
            if r.status_code in (429, 418) or r.status_code >= 500:
                retry_after = float(r.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, backoff))
                backoff = min(backoff * 1.8, 20.0)
                continue
            r.raise_for_status()
            data = r.json()
            
            out: List[FundingRate] = []
            for row in data:
                # Response format: {"symbol": "BTCUSDT", "fundingTime": 1640995200000, "fundingRate": "0.00010000"}
                out.append(
                    FundingRate(
                        timestamp=int(row["fundingTime"]),
                        funding_rate=float(row["fundingRate"]),
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


def iter_all_funding_rates(
    symbol: str,
    start_ms: int,
    end_ms: int,
    throttle_s: float = 0.2,
    session: Optional[requests.Session] = None,
) -> Iterable[FundingRate]:
    """
    Iterate through all funding rates between [start_ms, end_ms).
    
    Funding rates are published every 8 hours, so stepping is predictable.
    """
    sess = session or requests.Session()
    
    # Funding interval: 8 hours = 28800000 ms
    FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000
    
    t = start_ms
    last_ts: Optional[int] = None

    while t < end_ms:
        batch = fetch_funding_rates(sess, symbol, t, end_ms, limit=1000)
        if not batch:
            # No data; step forward by one funding period
            t += FUNDING_INTERVAL_MS
            time.sleep(throttle_s)
            continue

        for fr in batch:
            # Guard against duplicates
            if last_ts is not None and fr.timestamp <= last_ts:
                continue
            last_ts = fr.timestamp
            yield fr

        # Next start time = last timestamp + 1 ms to avoid overlap
        t = batch[-1].timestamp + 1
        time.sleep(throttle_s)


def write_csv_gz(path: str, rows: Iterable[FundingRate]) -> int:
    """
    Write funding rates to gzipped CSV with columns:
    timestamp_ms,funding_rate,symbol
    """
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "funding_rate", "symbol"])
        for fr in rows:
            w.writerow([fr.timestamp, f"{fr.funding_rate:.8f}", fr.symbol])
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Binance futures funding rate history")
    ap.add_argument("--symbols", help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--symbol", help="Single symbol (backward compatibility)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--out", help="Legacy single-output path")
    ap.add_argument("--out_dir", default="data/raw/futures", help="Output directory (default: data/raw/futures)")
    ap.add_argument("--force", action="store_true", help="Re-download even if cached")
    ap.add_argument("--max-requests-per-second", type=float, default=2.0, help="Polite rate limit")
    ap.add_argument("--throttle", type=float, default=0.6, help="Sleep between requests (seconds, overrides rate limit if larger)")
    args = ap.parse_args()

    symbols_arg = args.symbols or args.symbol
    if not symbols_arg:
        raise SystemExit("Provide --symbols or --symbol")
    symbols = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ms = parse_date_utc(args.start)
    end_ms = parse_date_utc(args.end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    throttle_s = max(args.throttle, 1.0 / max(args.max_requests_per_second, 0.1))
    session = requests.Session()

    for symbol in symbols:
        out_path = Path(args.out) if args.out and len(symbols) == 1 else out_dir / f"{symbol}_funding.csv.gz"
        if out_path.exists() and not args.force:
            print(f"[cache] {out_path} exists; skip (use --force to refresh)")
            continue
        print(f"Downloading funding rates for {symbol} from {args.start} to {args.end}")
        rows = iter_all_funding_rates(symbol, start_ms, end_ms, throttle_s=throttle_s, session=session)
        n = write_csv_gz(str(out_path), rows)
        print(f"Wrote {n} funding rate records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
