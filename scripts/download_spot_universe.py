"""Checksum-verified historical spot cohort, preserving absences and token identity."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import requests

from scripts.download_binance_archive import BASE_URL
from scripts.download_external_signals import _get
from scripts.download_spot_flow import BEGIN, END, EXTRA_COLUMNS, validate_bars

COHORT = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT",
          "TERRA_CLASSIC", "DOTUSDT", "AVAXUSDT", "DOGEUSDT", "SHIBUSDT",
          "POLYGON", "UNIUSDT", "LTCUSDT", "LINKUSDT")
COLUMNS = ("open", "high", "low", "close", "volume", *EXTRA_COLUMNS)
SNAPSHOT = "https://coinmarketcap.com/historical/20211226/"
POLYGON_NOTICE = pd.Timestamp("2024-08-28 09:00", tz="UTC")
IDENTITY_SOURCES = {
    "old_luna_halt": "https://www.binance.com/en/support/announcement/detail/514e0ba636e843bda47d5d740b7dadf4",
    "new_terra_is_other_token": "https://www.binance.com/en/support/announcement/detail/d044a6742e484b77a170111460b0eed3",
    "lunc_usdt_listing": "https://www.binance.com/en/support/announcement/detail/1e237033f63945d6b6acaf39f058d152",
    "polygon_notice": "https://www.binance.com/en/support/announcement/detail/6a6de383727f4659a3050f7982e1620f",
}


def source_periods(asset):
    """Research identifiers are never live order symbols; quantity stays 1:1."""
    if asset == "TERRA_CLASSIC":
        return (("LUNAUSDT", BEGIN, pd.Timestamp("2022-05-13 00:40", tz="UTC")),
                ("LUNCUSDT", pd.Timestamp("2022-09-09 08:00", tz="UTC"), END))
    if asset == "POLYGON":
        return (("MATICUSDT", BEGIN, pd.Timestamp("2024-09-10 03:00", tz="UTC")),
                ("POLUSDT", pd.Timestamp("2024-09-13 10:00", tz="UTC"), END))
    if asset not in COHORT:
        raise ValueError("Asset outside the fixed historical cohort")
    return ((asset, BEGIN, END),)


def _read_archive(payload, stem):
    member = stem + ".csv"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != [member] or archive.getinfo(member).file_size > 10_000_000:
            raise ValueError("Unexpected or oversized spot archive member")
        raw = pd.read_csv(archive.open(member), header=None)
    if raw.shape[1] != 12 or raw.empty:
        raise ValueError("Expected nonempty 12-column spot archive")
    numeric = raw.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Nonfinite published spot source fields")
    return numeric


def parse_partial_archive(payload, symbol, month, *, confirmed_duplicates=()):
    """Validate actual bars; only externally corroborated identical duplicates may collapse."""
    numeric = _read_archive(payload, f"{symbol}-4h-{month}")
    duplicated = numeric.iloc[:, 0].duplicated(keep=False)
    duplicate_ids = set(numeric.loc[duplicated, 0].tolist())
    if duplicate_ids != set(confirmed_duplicates):
        raise ValueError("Unverified duplicate spot timestamps")
    for _, group in numeric.loc[duplicated].groupby(0):
        if not group.eq(group.iloc[0]).all().all():
            raise ValueError("Conflicting duplicate spot values")
    numeric = numeric.loc[~numeric.iloc[:, 0].duplicated()].reset_index(drop=True)
    unit = "us" if month >= "2025-01" else "ms"
    times = pd.DatetimeIndex(pd.to_datetime(numeric.iloc[:, 0], unit=unit, utc=True)).as_unit("ns")
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    grid = pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="4h", inclusive="left").as_unit("ns")
    if not (times.is_unique and times.is_monotonic_increasing and times.isin(grid).all()):
        raise ValueError("Duplicate, unordered, off-grid or out-of-month spot bars")
    close_times = pd.DatetimeIndex(pd.to_datetime(numeric.iloc[:, 6], unit=unit, utc=True)).as_unit("ns")
    expected_close = times + pd.Timedelta("4h") - pd.Timedelta(1, unit=unit)
    halts = {("LUNAUSDT", "2022-05"): pd.Timestamp("2022-05-13 00:40", tz="UTC"),
             ("MATICUSDT", "2024-09"): pd.Timestamp("2024-09-10 03:00", tz="UTC")}
    if (symbol, month) in halts:
        halt = halts[(symbol, month)]
        expected_close = expected_close.where(times != halt.floor("4h"), halt - pd.Timedelta(1, unit=unit))
    if not close_times.equals(expected_close):
        raise ValueError(f"Unexpected spot candle close timestamp: {symbol} {month}")
    frame = pd.DataFrame({"timestamp": times})
    for name, offset in zip(COLUMNS, (1, 2, 3, 4, 5, 7, 8, 9, 10)):
        frame[name] = numeric.iloc[:, offset].to_numpy()
    validate_bars(frame)
    return frame


def fetch_month(symbol, month, cache, reuse_cache=None):
    stem = f"{symbol}-4h-{month}"
    url = f"{BASE_URL}/{symbol}/4h/{stem}.zip"

    def get(source):
        cached = reuse_cache / hashlib.sha256(source.encode()).hexdigest() if reuse_cache else None
        return cached.read_bytes() if cached is not None and cached.exists() else _get(source, cache)

    checksum = get(url + ".CHECKSUM").decode().split()
    if (len(checksum) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum[0])
            or checksum[1].lstrip("*") != stem + ".zip"):
        raise ValueError("Invalid published spot checksum")
    payload = get(url)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != checksum[0].lower():
        raise ValueError("Spot source SHA-256 mismatch")
    raw = _read_archive(payload, stem)
    duplicated = raw.iloc[:, 0].duplicated(keep=False)
    duplicate_ids = raw.loc[duplicated, 0].unique().tolist()
    repairs = []
    if duplicate_ids:
        unit = "us" if month >= "2025-01" else "ms"
        dates = pd.to_datetime(duplicate_ids, unit=unit, utc=True).strftime("%Y-%m-%d").unique()
        for date in dates:
            daily_stem = f"{symbol}-4h-{date}"
            daily_url = f"https://data.binance.vision/data/spot/daily/klines/{symbol}/4h/{daily_stem}.zip"
            proof = get(daily_url + ".CHECKSUM").decode().split()
            if (len(proof) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", proof[0])
                    or proof[1].lstrip("*") != daily_stem + ".zip"):
                raise ValueError("Invalid corroborating daily checksum")
            daily_payload = get(daily_url)
            daily_hash = hashlib.sha256(daily_payload).hexdigest()
            if daily_hash != proof[0].lower():
                raise ValueError("Corroborating daily SHA-256 mismatch")
            daily_raw = _read_archive(daily_payload, daily_stem)
            day_times = pd.to_datetime(raw.iloc[:, 0], unit=unit, utc=True).dt.strftime("%Y-%m-%d")
            monthly_day = raw.loc[day_times.eq(date)]
            for candidate in (monthly_day, daily_raw):
                for _, group in candidate.groupby(0):
                    if not group.eq(group.iloc[0]).all().all():
                        raise ValueError("Conflicting duplicate; study must stop")
            a = monthly_day.drop_duplicates().to_numpy()
            b = daily_raw.drop_duplicates().to_numpy()
            grid = pd.date_range(pd.Timestamp(date, tz="UTC"), periods=6, freq="4h").as_unit("ns")
            actual = pd.DatetimeIndex(pd.to_datetime(b[:, 0], unit=unit, utc=True)).as_unit("ns")
            if not np.array_equal(a, b) or not actual.equals(grid):
                raise ValueError("Daily source does not corroborate complete monthly day")
            repairs.append({"date": date, "url": daily_url, "sha256": daily_hash,
                "published_daily_rows": len(daily_raw), "canonical_day_rows": len(b),
                "monthly_duplicate_rows_removed": len(monthly_day) - len(a)})
    frame = parse_partial_archive(payload, symbol, month, confirmed_duplicates=duplicate_ids)
    return frame, {"symbol": symbol, "month": month, "url": url, "sha256": digest,
                   "published_rows": len(raw), "rows": len(frame), "duplicate_corroboration": repairs}


def missing_ranges(mask):
    """Compact exact absence ranges, with inclusive endpoints and counts."""
    positions = np.flatnonzero(mask.to_numpy())
    if not len(positions):
        return []
    groups = np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
    return [{"first": str(mask.index[g[0]]), "last": str(mask.index[g[-1]]), "bars": len(g)} for g in groups]


def assemble_asset(asset, results):
    pieces, archives = [], []
    for symbol, start, end in source_periods(asset):
        months = pd.date_range(start.normalize().replace(day=1), end - pd.Timedelta("1ns"), freq="MS")
        for month in months.strftime("%Y-%m"):
            frame, proof = results[(symbol, month)]
            keep = frame.loc[(frame.timestamp >= start) & (frame.timestamp < end)].copy()
            keep["market_symbol"] = symbol
            pieces.append(keep)
            archives.append({**proof, "used_rows": len(keep), "identity_start": str(start), "identity_end": str(end)})
    observed = pd.concat(pieces, ignore_index=True).set_index("timestamp")
    if not observed.index.is_unique or not observed.index.is_monotonic_increasing:
        raise ValueError("Overlapping or out-of-order underlying-asset identities")
    grid = pd.date_range(BEGIN, END, freq="4h", inclusive="left").as_unit("ns")
    frame = observed.reindex(grid)
    frame.index.name = "timestamp"
    frame["market_symbol"] = frame.market_symbol.fillna("")
    return frame, archives


def download(out, reuse_cache):
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    proofs = {}
    # Establish source eligibility before model evaluation; never replace an excluded asset.
    for asset in COHORT:
        symbol = source_periods(asset)[0][0]
        frame, proof = fetch_month(symbol, "2021-12", cache, reuse_cache)
        grid = pd.date_range("2021-12-01", "2022-01-01", freq="4h", inclusive="left", tz="UTC").as_unit("ns")
        if not pd.DatetimeIndex(frame.timestamp).equals(grid):
            raise ValueError("Cohort asset lacks complete December 2021 proof")
        proofs[asset] = proof
    cro_url = f"{BASE_URL}/CROUSDT/4h/CROUSDT-4h-2021-12.zip.CHECKSUM"
    try:
        _get(cro_url, cache)
    except requests.HTTPError as exc:
        if exc.response.status_code != 404:
            raise
    else:
        raise ValueError("CRO source eligibility differs from the fixed protocol; stop study")
    print("Verified 15 pre-period cohort archives; CRO source rule fails with HTTP 404", flush=True)
    tasks = []
    for asset in COHORT:
        for symbol, start, end in source_periods(asset):
            months = pd.date_range(start.normalize().replace(day=1), end - pd.Timedelta("1ns"), freq="MS")
            tasks.extend((symbol, month) for month in months.strftime("%Y-%m"))
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = {pool.submit(fetch_month, *task, cache, reuse_cache): task for task in tasks}
        for future in as_completed(pending):
            results[pending[future]] = future.result()
            if len(results) % 30 == 0:
                print(f"Verified broad spot {len(results)}/{len(tasks)} monthly archives", flush=True)
    manifest = {"source": "binance_spot_archive", "timeframe": "4h", "cohort": list(COHORT),
        "snapshot": SNAPSHOT, "snapshot_date": "2021-12-26", "retrieved_at": str(pd.Timestamp.now(tz="UTC")),
        "selection": "dated top20, exclude stablecoins/wrapped BTC, require December 2021 spot-USDT source",
        "exclusions": {"stable_or_wrapped": ["USDT", "USDC", "BUSD", "WBTC"],
                       "CRO": {"url": cro_url, "http_status": 404, "reason": "pre-period source rule"}},
        "eligibility_proofs": proofs, "identity_sources": IDENTITY_SOURCES,
        "polygon_notice": str(POLYGON_NOTICE), "gap_policy": "preserve_absence; no price filling",
        "missing_inventory_valuation": "zero sentinel marks; no execution; held-unquoted quality gate",
        "documentation": "https://github.com/binance/binance-public-data", "datasets": {}}
    for asset in COHORT:
        frame, archives = assemble_asset(asset, results)
        path = out / f"{asset}_4h.csv"
        path.write_text(frame.to_csv())
        absent = frame.open.isna()
        manifest["datasets"][asset] = {"file": path.name, "rows": len(frame),
            "observed_rows": int((~absent).sum()), "missing_rows": int(absent.sum()),
            "missing_ranges": missing_ranges(absent), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "archives": archives}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_universe(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest["source"] != "binance_spot_archive" or manifest["timeframe"] != "4h"
            or manifest["cohort"] != list(COHORT) or list(manifest["datasets"]) != list(COHORT)):
        raise ValueError("Manifest does not match fixed spot cohort/source")
    grid = pd.date_range(BEGIN, END, freq="4h", inclusive="left").as_unit("ns")
    markets = {}
    for asset, proof in manifest["datasets"].items():
        path = directory / proof["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != proof["sha256"]:
            raise ValueError("Historical spot CSV checksum mismatch")
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).astype("datetime64[ns, UTC]")
        frame = frame.set_index("timestamp")
        if not frame.index.equals(grid) or len(frame) != proof["rows"]:
            raise ValueError("Historical spot grid mismatch")
        present = frame.open.notna()
        validate_bars(frame.loc[present])
        if not frame.loc[~present, list(COLUMNS)].isna().all().all() or int((~present).sum()) != proof["missing_rows"]:
            raise ValueError("Missing quotes must have absent values in every source field")
        valid_identity = pd.Series(False, index=grid)
        for symbol, start, end in source_periods(asset):
            valid_identity |= (frame.market_symbol.eq(symbol) & (grid >= start) & (grid < end))
        if not valid_identity.loc[present].all() or frame.market_symbol.loc[~present].notna().any():
            raise ValueError("Wrong underlying token/market identity")
        markets[asset] = frame
    return markets, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/spot_universe"))
    parser.add_argument("--reuse-cache", type=Path, default=Path("data/raw/spot_flow/cache"))
    args = parser.parse_args()
    manifest = download(args.out, args.reuse_cache)
    print(f"Verified {sum(d['observed_rows'] for d in manifest['datasets'].values())} genuine spot bars", flush=True)


if __name__ == "__main__":
    main()
