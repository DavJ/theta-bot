"""Download complete spot kline months from Binance's public data archive.

Verify each published SHA-256 checksum. No exchange account/API key is used.
Source: https://github.com/binance/binance-public-data
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import pandas as pd
import requests


BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"


def parse_archive(payload: bytes, symbol: str, timeframe: str, month: str, *, allow_gaps=False) -> pd.DataFrame:
    expected_name = f"{symbol}-{timeframe}-{month}.csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if names != [expected_name] or archive.getinfo(expected_name).file_size > 10_000_000:
            raise ValueError("Unexpected archive member or oversized CSV.")
        with archive.open(expected_name) as csv:
            frame = pd.read_csv(csv, header=None)
    if frame.shape[1] != 12:
        raise ValueError("Expected 12 Binance kline columns.")
    frame = frame.iloc[:, :6].copy()
    frame.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    # Binance switched spot archive timestamps to microseconds in January 2025.
    unit = "us" if month >= "2025-01" else "ms"
    frame["timestamp"] = pd.to_datetime(pd.to_numeric(frame.timestamp), unit=unit, utc=True)
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    expected = pd.date_range(start, end, freq=timeframe, inclusive="left")
    actual = pd.DatetimeIndex(frame.timestamp)
    valid_subset = (len(actual) > 0 and actual.is_unique and actual.is_monotonic_increasing
                    and actual.isin(expected).all() and len(expected) - len(actual) <= 24)
    if not actual.equals(expected) and not (allow_gaps and valid_subset):
        raise ValueError(f"Archive {month} contains missing, duplicate, out-of-order or out-of-month candles.")
    return frame


def fetch_month(symbol, timeframe, month, *, allow_gaps=False):
    filename = f"{symbol}-{timeframe}-{month}.zip"
    url = f"{BASE_URL}/{symbol}/{timeframe}/{filename}"
    checksum = requests.get(url + ".CHECKSUM", timeout=20)
    checksum.raise_for_status()
    expected_hash = checksum.text.split()[0]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise ValueError(f"Invalid published checksum for {month}")
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    actual = hashlib.sha256(response.content).hexdigest()
    if actual != expected_hash.lower():
        raise ValueError(f"Archive checksum mismatch for {month}")
    frame = parse_archive(response.content, symbol, timeframe, month, allow_gaps=allow_gaps)
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    expected = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq=timeframe, inclusive="left")
    missing = expected.difference(pd.DatetimeIndex(frame.timestamp))
    return frame, {"month": month, "url": url, "sha256": actual, "rows": len(frame),
                   "missing_timestamps": missing.astype(str).tolist()}


def download(symbol, timeframe, start_month, end_month, out, *, allow_gaps=False):
    if not re.fullmatch(r"[A-Z0-9]{4,20}", symbol):
        raise ValueError("Symbol must be an uppercase Binance spot symbol.")
    if timeframe not in {"1h", "4h", "1d"}:
        raise ValueError("Supported archive timeframes: 1h, 4h, 1d.")
    for month in (start_month, end_month):
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise ValueError("Use YYYY-MM months.")
    start = pd.Timestamp(f"{start_month}-01", tz="UTC")
    end = pd.Timestamp(f"{end_month}-01", tz="UTC")
    if start > end or end >= pd.Timestamp.now(tz="UTC").normalize().replace(day=1):
        raise ValueError("Choose an ordered range of complete past months.")
    months = pd.date_range(start, end, freq="MS").strftime("%Y-%m").tolist()
    if len(months) > 24:
        raise ValueError("Download at most 24 months per request.")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda month: fetch_month(symbol, timeframe, month, allow_gaps=allow_gaps), months))
    frame = pd.concat([item[0] for item in results], ignore_index=True)
    output = frame.to_csv(index=False).encode()
    manifest = {
        "source": "binance_archive", "symbol": symbol, "timeframe": timeframe,
        "documentation": "https://github.com/binance/binance-public-data",
        "archives": [item[1] for item in results],
        "dataset_sha256": hashlib.sha256(output).hexdigest(),
        "rows": len(frame), "start": str(frame.timestamp.iloc[0]), "end": str(frame.timestamp.iloc[-1]),
        "gap_policy": "preserve_and_report" if allow_gaps else "reject",
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(output)
    out.with_suffix(".source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    end = pd.Timestamp.now(tz="UTC").normalize().replace(day=1) - pd.offsets.MonthBegin(1)
    start = end - pd.offsets.MonthBegin(5)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", choices=["1h", "4h", "1d"], default="1h")
    parser.add_argument("--start-month", default=start.strftime("%Y-%m"))
    parser.add_argument("--end-month", default=end.strftime("%Y-%m"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/BTCUSDT_1h_recent.csv"))
    parser.add_argument("--allow-gaps", action="store_true",
                        help="Research only: preserve and report up to 24 missing candles per month; never fill them")
    args = parser.parse_args()
    try:
        manifest = download(args.symbol, args.timeframe, args.start_month, args.end_month, args.out,
                            allow_gaps=args.allow_gaps)
    except (ValueError, requests.RequestException, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(f"Verified {len(manifest['archives'])} archives; {manifest['rows']} closed candles; {args.out}")


if __name__ == "__main__":
    main()
