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
import io
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional
from zipfile import ZipFile

import pandas as pd
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
            if r.status_code in (429, 418) or r.status_code >= 500:
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
    session: Optional[requests.Session] = None,
) -> Iterable[OpenInterest]:
    """
    Iterate through all open interest records between [start_ms, end_ms).
    """
    period_ms = period_to_ms(period)
    sess = session or requests.Session()
    
    t = start_ms
    last_ts: Optional[int] = None

    while t < end_ms:
        batch = fetch_open_interest_hist(sess, symbol, period, t, end_ms, limit=500)
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


def download_bulk_open_interest(
    symbol: str,
    start_ms: int,
    end_ms: int,
    session: requests.Session,
    throttle_s: float,
) -> Optional[pd.DataFrame]:
    """
    Attempt to download daily open interest archives from data.binance.vision.
    """
    base_url = "https://data.binance.vision"
    start_date = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).date()
    end_date = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).date()
    frames: List[pd.DataFrame] = []
    backoff = 1.0

    cur = start_date
    while cur <= end_date:
        daily = cur.isoformat()
        url = f"{base_url}/data/futures/um/daily/openInterest/{symbol}/{symbol}-open-interest-{daily}.zip"
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException:
            break
        if resp.status_code == 404:
            cur += timedelta(days=1)
            continue
        if resp.status_code in (429, 418) or resp.status_code >= 500:
            sleep_for = max(backoff, float(resp.headers.get("Retry-After", "1")))
            time.sleep(sleep_for)
            backoff = min(backoff * 1.8, 20.0)
            continue
        resp.raise_for_status()
        with ZipFile(io.BytesIO(resp.content)) as zf:
            if not zf.namelist():
                cur += timedelta(days=1)
                continue
            with zf.open(zf.namelist()[0]) as fh:
                frames.append(pd.read_csv(fh))
        cur += timedelta(days=1)
        time.sleep(throttle_s)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def normalize_oi_dataframe(df: pd.DataFrame, symbol: str) -> List[OpenInterest]:
    """
    Normalize a dataframe from API or bulk download into OpenInterest objects.
    """
    rename_map = {
        "sumOpenInterest": "open_interest",
        "sumOpenInterestValue": "open_interest_value",
        "openInterest": "open_interest",
        "openInterestValue": "open_interest_value",
    }
    df = df.rename(columns=rename_map)
    if "timestamp" not in df.columns and "timestamp_ms" in df.columns:
        df["timestamp"] = df["timestamp_ms"]
    if "timestamp" not in df.columns:
        return []

    df = df.dropna(subset=["timestamp"])
    rows: List[OpenInterest] = []
    for _, row in df.iterrows():
        try:
            ts = int(row["timestamp"])
            oi_val = float(row.get("open_interest", row.get("sumOpenInterest", row.get("openInterest", 0.0))))
            oi_value = float(
                row.get(
                    "open_interest_value",
                    row.get("sumOpenInterestValue", row.get("openInterestValue", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue
        rows.append(
            OpenInterest(
                timestamp=ts,
                sum_open_interest=oi_val,
                sum_open_interest_value=oi_value,
                symbol=symbol,
            )
        )
    return rows


def write_csv_gz(path: str, rows: Iterable[OpenInterest]) -> int:
    """
    Write open interest records to gzipped CSV with columns:
    timestamp_ms,open_interest,open_interest_value,symbol
    """
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "open_interest", "open_interest_value", "symbol"])
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
    ap = argparse.ArgumentParser(description="Download Binance futures open interest history")
    ap.add_argument("--symbols", help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--symbol", help="Single symbol (backward compatibility)")
    ap.add_argument("--period", default="1h", help="e.g. 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d (default: 1h)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--out", help="Legacy single-output path")
    ap.add_argument("--out_dir", default="data/raw/futures", help="Output directory (default: data/raw/futures)")
    ap.add_argument("--force", action="store_true", help="Re-download even if cached")
    ap.add_argument("--max-requests-per-second", type=float, default=2.0, help="Polite rate limit")
    ap.add_argument("--throttle", type=float, default=0.6, help="Sleep between requests (seconds, overrides rate limit if larger)")
    ap.add_argument("--prefer-bulk", action="store_true", help="Use data.binance.vision bulk files first")
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
        out_path = Path(args.out) if args.out and len(symbols) == 1 else out_dir / f"{symbol}_oi.csv.gz"
        if out_path.exists() and not args.force:
            print(f"[cache] {out_path} exists; skip (use --force to refresh)")
            continue

        print(f"Downloading open interest history for {symbol} (period={args.period}) from {args.start} to {args.end}")

        records: List[OpenInterest] = []
        if not args.prefer_bulk:
            records = list(iter_all_open_interest(symbol, args.period, start_ms, end_ms, throttle_s=throttle_s, session=session))

        if not records:
            print("API returned no data; attempting bulk archive fallback...", file=sys.stderr)
            bulk_df = download_bulk_open_interest(symbol, start_ms, end_ms, session=session, throttle_s=throttle_s)
            if bulk_df is not None:
                records = normalize_oi_dataframe(bulk_df, symbol)

        if not records:
            print(f"WARNING: No open interest data retrieved for {symbol}")
            continue

        n = write_csv_gz(str(out_path), records)
        print(f"Wrote {n} open interest records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
