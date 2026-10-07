"""Download checksum-verified spot 4h OHLCV and executed taker-buy flow only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from scripts.download_binance_archive import BASE_URL, parse_archive
from scripts.download_external_signals import _get

SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
BEGIN = pd.Timestamp("2022-01-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")
EXTRA_COLUMNS = ("quote_volume", "trade_count", "taker_buy_base", "taker_buy_quote")


def validate_bars(frame):
    values = frame[["open", "high", "low", "close", "volume", *EXTRA_COLUMNS]]
    if (not np.isfinite(values.to_numpy()).all()
            or (values.iloc[:, :4] <= 0).any().any() or (values.iloc[:, 4:] < 0).any().any()
            or (frame.low > frame[["open", "close"]].min(axis=1)).any()
            or (frame.high < frame[["open", "close"]].max(axis=1)).any()
            or (frame.taker_buy_base > frame.volume + 1e-8).any()
            or (frame.taker_buy_quote > frame.quote_volume + 1e-8).any()
            or (frame.trade_count % 1 != 0).any()):
        raise ValueError("Invalid spot OHLCV or executed taker-buy volume")


def parse_flow_archive(payload, symbol, month):
    frame = parse_archive(payload, symbol, "4h", month)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        raw = pd.read_csv(archive.open(f"{symbol}-4h-{month}.csv"), header=None)
    for col, offset in zip(EXTRA_COLUMNS, (7, 8, 9, 10)):
        frame[col] = pd.to_numeric(raw.iloc[:, offset]).to_numpy()
    unit = "us" if month >= "2025-01" else "ms"
    frame["timestamp"] = frame.timestamp.astype("datetime64[ns, UTC]")
    actual_close = pd.DatetimeIndex(pd.to_datetime(pd.to_numeric(raw.iloc[:, 6]), unit=unit, utc=True))
    expected_close = pd.DatetimeIndex(frame.timestamp) + pd.Timedelta("4h") - pd.Timedelta(1, unit=unit)
    if not actual_close.equals(expected_close):
        raise ValueError("Wrong candle close timestamp")
    validate_bars(frame)
    return frame


def fetch_flow_month(symbol, month, cache):
    stem = f"{symbol}-4h-{month}"
    url = f"{BASE_URL}/{symbol}/4h/{stem}.zip"
    checksum = _get(url + ".CHECKSUM", cache).decode().split()
    if (len(checksum) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum[0])
            or checksum[1].lstrip("*") != stem + ".zip"):
        raise ValueError("Invalid published spot archive checksum")
    payload = _get(url, cache)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != checksum[0].lower():
        raise ValueError("Spot archive checksum mismatch")
    frame = parse_flow_archive(payload, symbol, month)
    return frame, {"month": month, "url": url, "sha256": digest, "rows": len(frame)}


def download(out):
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    months = pd.date_range(BEGIN, END, freq="MS", inclusive="left").strftime("%Y-%m").tolist()
    tasks = [(symbol, month) for symbol in SYMBOLS for month in months]
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for task, result in zip(tasks, pool.map(lambda t: fetch_flow_month(*t, cache), tasks)):
            results[task] = result
            if len(results) % 30 == 0:
                print(f"Verified spot flow {len(results)}/{len(tasks)} archives", flush=True)
    expected = pd.date_range(BEGIN, END, freq="4h", inclusive="left")
    manifest = {"source": "binance_spot_archive", "timeframe": "4h",
                "retrieved_at": str(pd.Timestamp.now(tz="UTC")), "gap_policy": "reject",
                "documentation": "https://github.com/binance/binance-public-data", "datasets": {}}
    for symbol in SYMBOLS:
        frame = pd.concat([results[(symbol, m)][0] for m in months], ignore_index=True)
        if not pd.DatetimeIndex(frame.timestamp).equals(expected):
            raise ValueError("Incomplete or overlapping spot 4h history; no gap filling allowed")
        path = out / f"{symbol}_4h.csv"
        path.write_text(frame.to_csv(index=False))
        manifest["datasets"][symbol] = {"file": path.name, "rows": len(frame),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "first": str(frame.timestamp.iloc[0]), "last": str(frame.timestamp.iloc[-1]),
            "archives": [results[(symbol, m)][1] for m in months]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/spot_flow"))
    args = parser.parse_args()
    result = download(args.out)
    print(f"Verified {sum(d['rows'] for d in result['datasets'].values())} spot 4h bars")


if __name__ == "__main__":
    main()
