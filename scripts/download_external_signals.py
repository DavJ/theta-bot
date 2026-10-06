"""Retrieve bounded public source snapshots for the fixed external-signal study."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import threading
import zipfile

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import requests

from scripts.download_binance_archive import parse_archive

START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2026-10-01", tz="UTC")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
HTTP = threading.local()


def fred_availability(dates, code):
    if code == "DCOILWTICO":
        return dates + pd.Timedelta("10D")
    offset = pd.offsets.CustomBusinessDay(n=2, calendar=USFederalHolidayCalendar())
    return pd.Series([date + offset for date in dates], index=dates.index)


def _get(url, cache):
    path = cache / hashlib.sha256(url.encode()).hexdigest()
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            if not hasattr(HTTP, "session"):
                HTTP.session = requests.Session()
            response = HTTP.session.get(url, timeout=25)
            response.raise_for_status()
            if len(response.content) > 8_000_000:
                raise ValueError("Source payload exceeds bounded download size")
            temporary = path.with_suffix(".partial")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def _csv_members(payload):
    if payload[:2] != b"PK":
        return [pd.read_csv(io.BytesIO(payload))]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if any(i.file_size > 12_000_000 for i in archive.infolist()):
            raise ValueError("Oversized uncompressed source")
        return [pd.read_csv(io.BytesIO(archive.read(n)))
                for n in archive.namelist() if n.endswith(".csv")]


def _records(source, dates, values, availability):
    frame = pd.DataFrame({"source": source, "event_time": dates,
                          "available_at": availability, "value": values})
    frame = frame.loc[(frame.event_time >= START) & (frame.event_time < END)].dropna(subset=["value"])
    return frame.sort_values("event_time").reset_index(drop=True)


def _archive(kind, symbol, month, cache):
    stem = f"{symbol}-1d-{month}" if kind == "spot" else f"{symbol}-fundingRate-{month}"
    base = (f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1d" if kind == "spot"
            else f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}")
    url = f"{base}/{stem}.zip"
    checksum = _get(url + ".CHECKSUM", cache).decode().split()[0]
    payload = _get(url, cache)
    actual = hashlib.sha256(payload).hexdigest()
    if checksum.lower() != actual:
        raise ValueError(f"Checksum mismatch: {url}")
    if kind == "spot":
        prices = parse_archive(payload, symbol, "1d", month)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if archive.namelist() != [stem + ".csv"]:
                raise ValueError("Unexpected spot member")
            raw = pd.read_csv(archive.open(stem + ".csv"), header=None)
        volume = pd.to_numeric(raw.iloc[:, 5])
        buy = pd.to_numeric(raw.iloc[:, 9])
        if ((volume <= 0).any() or (buy < 0).any() or (buy > volume).any()):
            raise ValueError("Invalid daily taker volume")
        frame = prices.assign(taker_buy_share=(buy / volume).to_numpy())
    else:
        members = _csv_members(payload)
        if len(members) != 1 or set(members[0]) != {"calc_time", "funding_interval_hours", "last_funding_rate"}:
            raise ValueError("Unexpected funding schema")
        frame = members[0]
        frame["event_time"] = pd.to_datetime(frame.calc_time, unit="ms", utc=True)
        if (not frame.event_time.is_unique or not frame.event_time.is_monotonic_increasing
                or (frame.event_time.dt.strftime("%Y-%m") != month).any()
                or not np.isfinite(frame.last_funding_rate).all()
                or (frame.last_funding_rate.abs() > .5).any()
                or (frame.funding_interval_hours <= 0).any()):
            raise ValueError("Invalid funding records")
    return frame, {"url": url, "sha256": actual, "rows": len(frame)}


def download(out):
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    sources, provenance, spots = [], [], {}
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM,VIXCLS,DCOILWTICO,DGS10"
    payload = _get(fred_url, cache)
    columns = set()
    for table in _csv_members(payload):
        dates = pd.to_datetime(table.observation_date, utc=True)
        for code in table.columns.drop("observation_date"):
            if code not in {"NASDAQCOM", "VIXCLS", "DCOILWTICO", "DGS10"} or code in columns:
                raise ValueError("Unexpected or repeated FRED series")
            columns.add(code)
            keep = (dates >= START) & (dates < END)
            selected_dates = dates.loc[keep]
            sources.append(_records(code, selected_dates, pd.to_numeric(table.loc[keep, code], errors="coerce"),
                                    fred_availability(selected_dates, code)))
    if columns != {"NASDAQCOM", "VIXCLS", "DCOILWTICO", "DGS10"}:
        raise ValueError("Missing FRED series")
    provenance.append({"source": "FRED", "url": fred_url, "raw_sha256": hashlib.sha256(payload).hexdigest(),
                       "vintage": "latest_snapshot_not_first_release",
                       "availability": "assumed lag: oil 10 calendar days; others 2 US federal business days"})
    ecb_url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
    payload = _get(ecb_url, cache)
    table, = _csv_members(payload)
    dates = pd.to_datetime(table.Date, utc=True)
    for code, values in {"EURUSD": table.USD, "USDJPY": table.JPY / table.USD}.items():
        sources.append(_records(code, dates, values, dates + pd.Timedelta(hours=18)))
    provenance.append({"source": "ECB", "url": ecb_url, "raw_sha256": hashlib.sha256(payload).hexdigest(),
                       "vintage": "current_reference_snapshot", "availability": "observation date +18h UTC"})
    fng_url = "https://api.alternative.me/fng/?limit=0&format=json"
    payload = _get(fng_url, cache)
    result = json.loads(payload)
    if result.get("metadata", {}).get("error") is not None:
        raise ValueError("Fear & Greed provider error")
    table = pd.DataFrame(result["data"])
    dates = pd.to_datetime(pd.to_numeric(table.timestamp), unit="s", utc=True)
    sources.append(_records("FEAR_GREED", dates, pd.to_numeric(table.value), dates + pd.Timedelta("1D")))
    provenance.append({"source": "Alternative.me Fear & Greed", "url": fng_url,
                       "attribution": "https://alternative.me/crypto/fear-and-greed-index/",
                       "raw_sha256": hashlib.sha256(payload).hexdigest(), "availability": "index timestamp +1d",
                       "vintage": "historical_endpoint_snapshot; not independent NLP"})
    begin = pd.Timestamp("2022-01-01", tz="UTC")
    months = pd.date_range(begin, END, freq="MS", inclusive="left").strftime("%Y-%m")
    tasks = [(kind, symbol, month) for symbol in SYMBOLS for kind in ("spot", "funding") for month in months]
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for task, result in zip(tasks, pool.map(lambda t: _archive(*t, cache), tasks)):
            results[task] = result
            if len(results) % 30 == 0:
                print(f"Verified {len(results)}/{len(tasks)} archives", flush=True)
    for symbol in SYMBOLS:
        spot = pd.concat([results[("spot", symbol, month)][0] for month in months], ignore_index=True)
        spot.to_csv(out / f"{symbol}_flow.csv", index=False)
        spots[symbol] = {"file": f"{symbol}_flow.csv", "rows": len(spot),
                          "sha256": hashlib.sha256((out / f"{symbol}_flow.csv").read_bytes()).hexdigest()}
        sources.append(_records(f"FLOW_{symbol}", spot.timestamp, spot.taker_buy_share,
                                spot.timestamp + pd.Timedelta("1D")))
        funding = pd.concat([results[("funding", symbol, month)][0] for month in months], ignore_index=True)
        funding["day"] = funding.event_time.dt.floor("D")
        grouped = funding.groupby("day").agg(value=("last_funding_rate", "sum"),
                    available_at=("event_time", "max"), hours=("funding_interval_hours", "sum"))
        expected = pd.date_range(begin, END, freq="D", inclusive="left")
        if not grouped.index.equals(expected) or (grouped.hours < 23.5).any():
            raise ValueError(f"Funding daily coverage incomplete for {symbol}")
        sources.append(_records(f"FUNDING_{symbol}", grouped.index, grouped.value.to_numpy(),
                                grouped.available_at.to_numpy() + pd.Timedelta("5min")))
    all_sources = pd.concat(sources, ignore_index=True).sort_values(["source", "event_time"])
    if (all_sources.duplicated(["source", "event_time"]).any()
            or not np.isfinite(all_sources.value).all()
            or (all_sources.available_at < all_sources.event_time).any()):
        raise ValueError("Invalid combined signal grain or availability")
    all_sources.to_csv(out / "signals.csv", index=False)
    profile = {code: {"rows": len(f), "first": str(f.event_time.min()), "last": str(f.event_time.max()),
                       "min": float(f.value.min()), "max": float(f.value.max())}
               for code, f in all_sources.groupby("source")}
    manifest = {"retrieved_at": str(pd.Timestamp.now(tz="UTC")), "profile": profile,
                "signals_sha256": hashlib.sha256((out / "signals.csv").read_bytes()).hexdigest(),
                "spot_files": spots, "provenance": provenance,
                "archives": [{"kind": k, "symbol": s, "month": m, **r[1]}
                             for (k, s, m), r in results.items()],
                "historical_first_release_vintages_verified": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/external_signals"))
    args = parser.parse_args()
    manifest = download(args.out)
    print(f"Completed: {len(manifest['profile'])} series, {len(manifest['archives'])} verified archives")


if __name__ == "__main__":
    main()
