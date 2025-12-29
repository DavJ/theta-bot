from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


TS_CANDIDATES = ("close_time_ms", "timestamp_ms", "timestamp", "open_time_ms")


def _pick_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _read_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    ts_col = _pick_column(df, TS_CANDIDATES)
    if ts_col is None:
        raise ValueError(f"No timestamp column in {path}")
    df = df.rename(columns={ts_col: "timestamp_ms"})
    df = df.sort_values("timestamp_ms")
    df.index = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df


def load_symbol_panel(
    symbol: str,
    interval: str = "1h",
    data_dir: str = "data/raw",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    base = Path(data_dir)
    spot = _read_series(base / "spot" / f"{symbol}_{interval}.csv.gz")
    mark = _read_series(base / "futures" / f"{symbol}_mark_{interval}.csv.gz")
    funding = _read_series(base / "futures" / f"{symbol}_funding.csv.gz")
    oi = _read_series(base / "futures" / f"{symbol}_oi.csv.gz")
    basis_path = base / "futures" / f"{symbol}_basis.csv.gz"
    basis = _read_series(basis_path) if basis_path.exists() else None

    # Normalize column names
    if "close" not in spot.columns:
        close_col = _pick_column(spot, ["spot_close", "price"])
        if close_col:
            spot = spot.rename(columns={close_col: "close"})
    if "close" not in mark.columns:
        mclose = _pick_column(mark, ["mark_close", "price"])
        if mclose:
            mark = mark.rename(columns={mclose: "close"})
    fund_col = _pick_column(funding, ["funding_rate", "fundingRate"])
    if fund_col and fund_col != "funding_rate":
        funding = funding.rename(columns={fund_col: "funding_rate"})
    oi_col = _pick_column(oi, ["open_interest", "sumOpenInterest"])
    if oi_col and oi_col != "open_interest":
        oi = oi.rename(columns={oi_col: "open_interest"})
    if basis is not None and "basis" not in basis.columns:
        bcol = _pick_column(basis, ["value", "basis_value"])
        if bcol:
            basis = basis.rename(columns={bcol: "basis"})

    # Convert to hourly cadence
    funding_1h = funding[["funding_rate"]].resample("1h").ffill()
    oi_hourly = oi[["open_interest"]].resample("1h").last()
    oi_filled = oi_hourly.ffill()
    oi_filled["open_interest_is_filled"] = oi_hourly.isna() & oi_filled["open_interest"].notna()

    mark_close = mark[["close"]].rename(columns={"close": "mark_close"})
    spot_close = spot[["close"]].rename(columns={"close": "spot_close"})

    if basis is None:
        aligned = pd.concat([spot_close, mark_close], axis=1, join="inner")
        basis_series = np.log(aligned["mark_close"]) - np.log(aligned["spot_close"])
        basis = pd.DataFrame({"basis": basis_series})
    else:
        basis = basis[["basis"]]

    panel = pd.concat(
        [
            spot_close,
            mark_close,
            funding_1h,
            oi_filled,
            basis,
        ],
        axis=1,
        join="inner",
    ).dropna()

    if start:
        panel = panel.loc[start:]
    if end:
        panel = panel.loc[:end]

    panel["returns"] = np.log(panel["spot_close"]).diff()
    return panel


def save_processed(symbol: str, df: pd.DataFrame, out_dir: str) -> Path:
    out_path = Path(out_dir) / f"{symbol}_1h.csv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, compression="gzip")
    return out_path
