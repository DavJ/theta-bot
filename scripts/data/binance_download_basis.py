#!/usr/bin/env python3
"""
Download Binance futures basis history and save to CSV.GZ.

Basis = Futures Price - Index Price
This shows the premium/discount of futures relative to spot.

Example:
  python scripts/data/binance_download_basis.py \
    --symbol BTCUSDT \
    --period 5m \
    --start 2024-01-01 --end 2024-10-01 \
    --out data/raw/futures/BTCUSDT_basis.csv.gz

Endpoint: /futures/data/basis
Note: If this endpoint is unavailable, basis can be computed from mark_price - spot_price.

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
class Basis:
    timestamp: int  # time (ms)
    basis: float  # futures - index price
    basis_rate: float  # basis / index price
    annualized_basis_rate: float  # annualized rate
    symbol: str


def fetch_basis_hist(
    session: requests.Session,
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    contract_type: str = "PERPETUAL",
    limit: int = 500,
    timeout_s: int = 30,
    max_retries: int = 6,
) -> List[Basis]:
    """
    Fetch basis history from Binance futures API.
    
    Note: This endpoint may not be available for all symbols/periods.
    In such cases, compute basis from mark price - spot price.
    """
    url = f"{BINANCE_FAPI_BASE}/futures/data/basis"
    params = {
        "pair": symbol.upper(),
        "contractType": contract_type,
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
            
            # If endpoint doesn't exist or symbol not supported, return empty
            if r.status_code == 400 or r.status_code == 404:
                print(f"Warning: Basis endpoint not available for {symbol}. Consider computing from mark - spot.", file=sys.stderr)
                return []
            
            r.raise_for_status()
            data = r.json()
            
            out: List[Basis] = []
            for row in data:
                # Response format may include: timestamp, basis, basisRate, annualizedBasisRate
                # Actual format may vary; adapt as needed
                out.append(
                    Basis(
                        timestamp=int(row["timestamp"]),
                        basis=float(row.get("basis", 0.0)),
                        basis_rate=float(row.get("basisRate", 0.0)),
                        annualized_basis_rate=float(row.get("annualizedBasisRate", 0.0)),
                        symbol=symbol,
                    )
                )
            return out
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries - 1:
                # If final attempt fails, return empty to allow fallback
                print(f"Warning: Could not fetch basis data: {e}", file=sys.stderr)
                return []
            time.sleep(backoff)
            backoff = min(backoff * 1.8, 20.0)

    return []


def iter_all_basis(
    symbol: str,
    period: str,
    start_ms: int,
    end_ms: int,
    contract_type: str = "PERPETUAL",
    throttle_s: float = 0.2,
) -> Iterable[Basis]:
    """
    Iterate through all basis records between [start_ms, end_ms).
    """
    period_ms = period_to_ms(period)
    session = requests.Session()
    
    t = start_ms
    last_ts: Optional[int] = None

    while t < end_ms:
        batch = fetch_basis_hist(session, symbol, period, t, end_ms, contract_type, limit=500)
        if not batch:
            # No data; step forward
            t += period_ms
            time.sleep(throttle_s)
            continue

        for b in batch:
            if last_ts is not None and b.timestamp <= last_ts:
                continue
            last_ts = b.timestamp
            yield b

        # Next start time
        t = batch[-1].timestamp + 1
        time.sleep(throttle_s)


def write_csv_gz(path: str, rows: Iterable[Basis]) -> int:
    """
    Write basis records to gzipped CSV with columns:
    timestamp,basis,basisRate,annualizedBasisRate,symbol
    """
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "basis", "basisRate", "annualizedBasisRate", "symbol"])
        for b in rows:
            w.writerow([
                b.timestamp, 
                f"{b.basis:.8f}", 
                f"{b.basis_rate:.8f}",
                f"{b.annualized_basis_rate:.8f}",
                b.symbol
            ])
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download Binance futures basis history"
    )
    ap.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    ap.add_argument("--period", required=True, help="e.g. 5m, 15m, 30m, 1h")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--contract-type", default="PERPETUAL", help="Contract type (default: PERPETUAL)")
    ap.add_argument("--out", required=True, help="Output path, e.g. data/raw/futures/BTCUSDT_basis.csv.gz")
    ap.add_argument("--throttle", type=float, default=0.2, help="Sleep between requests (seconds)")
    args = ap.parse_args()

    start_ms = parse_date_utc(args.start)
    end_ms = parse_date_utc(args.end)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")

    print(f"Downloading basis history for {args.symbol} (period={args.period}, contract={args.contract_type})")
    print(f"Date range: {args.start} to {args.end}")
    
    rows = iter_all_basis(
        args.symbol, 
        args.period, 
        start_ms, 
        end_ms, 
        contract_type=args.contract_type,
        throttle_s=args.throttle
    )
    n = write_csv_gz(args.out, rows)
    
    if n == 0:
        print(f"WARNING: No basis data retrieved. Endpoint may not support {args.symbol}.")
        print(f"Consider computing basis from: mark_price - spot_price")
        return 1
    
    print(f"Wrote {n} basis records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
