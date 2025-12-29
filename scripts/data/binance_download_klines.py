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
from pathlib import Path
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
    open_time_ms: int
    close_time_ms: int
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
            if r.status_code in (429, 418) or r.status_code >= 500:
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
                        open_time_ms=int(row[0]),
                        close_time_ms=int(row[6]),
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
    session: Optional[requests.Session] = None,
) -> Iterable[Kline]:
    """
    Iterate through all klines between [start_ms, end_ms).
    """
    step_ms = interval_to_ms(interval)
    sess = session or requests.Session()

    t = start_ms
    last_ts: Optional[int] = None

    while t < end_ms:
        batch = fetch_klines(sess, symbol, interval, t, end_ms, limit=1000)
        if not batch:
            # No data returned; step forward by one interval to avoid infinite loops.
            t += step_ms
            time.sleep(throttle_s)
            continue

        for k in batch:
            # Guard against duplicates / non-monotonic returns
            if last_ts is not None and k.open_time_ms <= last_ts:
                continue
            last_ts = k.open_time_ms
            yield k

        # Next start time = last timestamp + interval
        t = batch[-1].open_time_ms + step_ms
        time.sleep(throttle_s)


def write_csv_gz(path: str, rows: Iterable[Kline]) -> int:
    """
    Write rows to a gzipped CSV with columns:
    open_time_ms,close_time_ms,open,high,low,close,volume
    """
    count = 0
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "open_time_ms",
                "close_time_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        )
        for k in rows:
            w.writerow(
                [
                    k.open_time_ms,
                    k.close_time_ms,
                    f"{k.open:.8f}",
                    f"{k.high:.8f}",
                    f"{k.low:.8f}",
                    f"{k.close:.8f}",
                    f"{k.volume:.8f}",
                ]
            )
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="Comma-separated list, e.g. BTCUSDT,ETHUSDT")
    ap.add_argument("--symbol", help="Single symbol (backward compatibility)")
    ap.add_argument("--interval", required=True, help="e.g. 1h, 15m, 1d")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or ISO datetime (UTC assumed if no tz)")
    ap.add_argument("--out", help="Legacy output path for single symbol")
    ap.add_argument("--out_dir", default="data/raw/spot", help="Output directory (default: data/raw/spot)")
    ap.add_argument("--force", action="store_true", help="Re-download even if output exists")
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
        out_path = Path(args.out) if args.out and len(symbols) == 1 else out_dir / f"{symbol}_{args.interval}.csv.gz"
        if out_path.exists() and not args.force:
            print(f"[cache] {out_path} exists; skip (use --force to refresh)")
            continue
        rows = iter_all_klines(symbol, args.interval, start_ms, end_ms, throttle_s=throttle_s, session=session)
        n = write_csv_gz(str(out_path), rows)
        print(f"{symbol}: wrote {n} klines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
