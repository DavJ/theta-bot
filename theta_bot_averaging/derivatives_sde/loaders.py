#!/usr/bin/env python3
"""Data loaders for derivatives SDE decomposition."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def load_csv_gz(path: Path) -> Optional[pd.DataFrame]:
    """Load gzipped CSV with a millisecond timestamp column."""
    if not path.exists():
        return None
    df = pd.read_csv(path, compression="gzip")
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing 'timestamp' column in {path}")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp"])
    return df


def load_spot(symbol: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """Load spot klines at 1h cadence."""
    path = Path(data_dir) / "spot" / f"{symbol}_1h.csv.gz"
    df = load_csv_gz(path)
    if df is None:
        raise FileNotFoundError(f"Spot data not found: {path}")
    if "close" not in df.columns:
        raise ValueError(f"Missing 'close' column in {path}")
    return df


def load_funding(symbol: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """Load funding series and forward-fill to 1h."""
    path = Path(data_dir) / "futures" / f"{symbol}_funding.csv.gz"
    df = load_csv_gz(path)
    if df is None:
        raise FileNotFoundError(f"Funding data not found: {path}")
    if "fundingRate" not in df.columns:
        raise ValueError(f"Missing 'fundingRate' column in {path}")
    return df[["fundingRate"]].resample("1h").ffill()


def load_oi(symbol: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """Load open interest series and resample to 1h using last observation."""
    path = Path(data_dir) / "futures" / f"{symbol}_oi.csv.gz"
    df = load_csv_gz(path)
    if df is None:
        raise FileNotFoundError(f"Open interest data not found: {path}")
    if "sumOpenInterest" not in df.columns:
        raise ValueError(f"Missing 'sumOpenInterest' column in {path}")
    return df[["sumOpenInterest"]].resample("1h").last()


def load_basis(symbol: str, data_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """Load basis series and resample to 1h if available."""
    path = Path(data_dir) / "futures" / f"{symbol}_basis.csv.gz"
    df = load_csv_gz(path)
    if df is None:
        return None
    if "basis" not in df.columns:
        raise ValueError(f"Missing 'basis' column in {path}")
    return df[["basis"]].resample("1h").last()


def align_indices(series_list: Iterable[pd.Series], how: str = "inner") -> pd.DatetimeIndex:
    """Align indices for a collection of series."""
    series_list = list(series_list)
    if not series_list:
        return pd.DatetimeIndex([])
    idx = series_list[0].index
    for s in series_list[1:]:
        idx = idx.intersection(s.index) if how == "inner" else idx.union(s.index)
    return idx.sort_values()
