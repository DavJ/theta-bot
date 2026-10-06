"""Verified perpetual trade/mark bars and original funding, without credentials."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from scripts.download_external_signals import _get, _archive, SYMBOLS, END

BEGIN = pd.Timestamp("2022-01-01", tz="UTC")


def parse_futures_archive(payload, symbol, timeframe, month, *, allow_partial=False):
    name = f"{symbol}-{timeframe}-{month}.csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [name] or archive.getinfo(name).file_size > 12_000_000:
            raise ValueError("Unexpected or oversized futures archive")
        frame = pd.read_csv(archive.open(name), header=None)
    if frame.shape[1] != 12:
        raise ValueError("Expected 12 futures kline fields")
    if str(frame.iloc[0, 0]).lower().replace("_", "").replace(" ", "") == "opentime":
        frame = frame.iloc[1:].reset_index(drop=True)
    frame = frame.iloc[:, :6].copy()
    frame.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    # Futures timestamps remain milliseconds; the spot 2025 switch does not apply.
    frame.timestamp = pd.to_datetime(pd.to_numeric(frame.timestamp), unit="ms", utc=True)
    for column in frame.columns.drop("timestamp"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    start = pd.Timestamp(month if len(month) == 10 else month + "-01", tz="UTC")
    end = start + (pd.Timedelta("1D") if len(month) == 10 else pd.offsets.MonthBegin())
    expected = pd.date_range(start, end, freq=timeframe, inclusive="left")
    actual = pd.DatetimeIndex(frame.timestamp)
    subset = (len(actual) > 0 and actual.is_unique and actual.is_monotonic_increasing
              and actual.isin(expected).all() and len(expected) - len(actual) <= 24)
    if not actual.equals(expected) and not (allow_partial and subset):
        raise ValueError(f"Missing, duplicate, unordered or out-of-period futures bar: {symbol} {month} {timeframe}")
    values = frame.iloc[:, 1:]
    if (not np.isfinite(values.to_numpy()).all() or (values.iloc[:, :4] <= 0).any().any()
            or (values.volume < 0).any() or (frame.low > frame[["open", "close"]].min(axis=1)).any()
            or (frame.high < frame[["open", "close"]].max(axis=1)).any()):
        raise ValueError("Invalid futures OHLCV")
    return frame


def fetch(task, cache):
    kind, symbol, month = task
    if kind == "funding":
        return _archive(kind, symbol, month, cache)
    timeframe, directory = ("1d", "klines") if kind == "trade" else ("1h", "markPriceKlines")
    url = f"https://data.binance.vision/data/futures/um/monthly/{directory}/{symbol}/{timeframe}/{symbol}-{timeframe}-{month}.zip"
    payload = _get(url, cache)
    actual = hashlib.sha256(payload).hexdigest()
    expected = _get(url + ".CHECKSUM", cache).decode().split()[0].lower()
    if actual != expected:
        raise ValueError("Futures checksum mismatch")
    frame = parse_futures_archive(payload, symbol, timeframe, month, allow_partial=True)
    start = pd.Timestamp(month + "-01", tz="UTC")
    grid = pd.date_range(start, start + pd.offsets.MonthBegin(), freq=timeframe, inclusive="left")
    missing = grid.difference(pd.DatetimeIndex(frame.timestamp))
    supplements = []
    for day in missing.floor("D").unique():
        date = day.strftime("%Y-%m-%d")
        daily_url = f"https://data.binance.vision/data/futures/um/daily/{directory}/{symbol}/{timeframe}/{symbol}-{timeframe}-{date}.zip"
        patch = _get(daily_url, cache)
        patch_hash = hashlib.sha256(patch).hexdigest()
        if patch_hash != _get(daily_url + ".CHECKSUM", cache).decode().split()[0].lower():
            raise ValueError("Daily futures repair checksum mismatch")
        daily = parse_futures_archive(patch, symbol, timeframe, date)
        overlap = frame.merge(daily, on="timestamp", suffixes=("_monthly", "_daily"))
        for column in ("open", "high", "low", "close", "volume"):
            np.testing.assert_allclose(overlap[column + "_monthly"], overlap[column + "_daily"], rtol=1e-12)
        frame = pd.concat([frame, daily.loc[daily.timestamp.isin(missing)]], ignore_index=True).sort_values("timestamp")
        supplements.append({"url": daily_url, "sha256": patch_hash, "rows": len(daily)})
    if not pd.DatetimeIndex(frame.timestamp).equals(grid):
        raise ValueError("Official daily repair did not restore complete futures grid")
    return frame.reset_index(drop=True), {"url": url, "sha256": actual, "rows": len(frame),
                                         "missing_monthly_bars": len(missing), "official_daily_repairs": supplements}


def download(out, cache):
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(BEGIN, END, freq="MS", inclusive="left").strftime("%Y-%m")
    tasks = [(kind, symbol, month) for symbol in SYMBOLS for kind in ("trade", "mark", "funding") for month in months]
    data = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for task, result in zip(tasks, pool.map(lambda t: fetch(t, cache), tasks)):
            data[task] = result
            if len(data) % 30 == 0:
                print(f"Verified {len(data)}/{len(tasks)} futures archives", flush=True)
    files = {}
    for symbol in SYMBOLS:
        files[symbol] = {}
        for kind in ("trade", "mark", "funding"):
            frame = pd.concat([data[kind, symbol, month][0] for month in months], ignore_index=True)
            if kind == "funding":
                frame = frame[["event_time", "last_funding_rate", "funding_interval_hours"]]
                if (not frame.event_time.is_unique or not frame.event_time.is_monotonic_increasing
                        or frame.event_time.min() < BEGIN or frame.event_time.max() >= END):
                    raise ValueError("Invalid combined funding events")
                hours = frame.groupby(frame.event_time.dt.floor("D")).funding_interval_hours.sum()
                expected = pd.date_range(BEGIN, END, freq="D", inclusive="left")
                if not hours.index.equals(expected) or (hours < 23.5).any():
                    raise ValueError("Incomplete actual funding coverage")
            filename = f"{symbol}_{kind}.csv"
            frame.to_csv(out / filename, index=False)
            files[symbol][kind] = {"file": filename, "rows": len(frame),
                "sha256": hashlib.sha256((out / filename).read_bytes()).hexdigest()}
    manifest = {"source": "binance_usdm_archive", "retrieved_at": str(pd.Timestamp.now(tz="UTC")),
                "files": files, "archives": [{"kind": k, "symbol": s, "month": m, **v[1]}
                                               for (k, s, m), v in data.items()],
                "funding_mark_precision": "original settlement rates/timestamps; charge bounded by hourly mark OHLC"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Completed: {len(tasks)} verified futures archives", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/futures_research"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw/external_signals/cache"))
    args = parser.parse_args()
    download(args.out, args.cache)


if __name__ == "__main__":
    main()
