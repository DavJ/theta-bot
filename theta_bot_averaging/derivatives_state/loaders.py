#!/usr/bin/env python3
"""
Data loaders for derivatives state module.

Load spot and futures series following DATA_PROTOCOL conventions:
- All data aligned to 1h grid
- UTC timestamps
- Forward-fill funding rates (8h -> 1h)
- Resample other series to 1h using last observation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv_gz(path: Path) -> Optional[pd.DataFrame]:
    """
    Load gzipped CSV and return DataFrame with DatetimeIndex.
    
    Parameters
    ----------
    path : Path
        Path to gzipped CSV file
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with UTC DatetimeIndex, or None if file doesn't exist
    """
    if not path.exists():
        return None
    
    try:
        df = pd.read_csv(path, compression="gzip")
        if "timestamp" not in df.columns:
            raise ValueError(f"No 'timestamp' column in {path}")
        
        # Convert timestamp to datetime index
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64")
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Convert to UTC DatetimeIndex
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp"])
        
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")


def load_spot_series(symbol: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load spot klines data for symbol.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT")
    data_dir : str
        Base data directory (default: "data/raw")
        
    Returns
    -------
    pd.DataFrame
        Spot klines with columns: open, high, low, close, volume
        Index: UTC DatetimeIndex at 1h intervals
    """
    path = Path(data_dir) / "spot" / f"{symbol}_1h.csv.gz"
    df = load_csv_gz(path)
    
    if df is None:
        raise FileNotFoundError(f"Spot data not found: {path}")
    
    required_cols = ["close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in spot data: {missing}")
    
    return df


def load_funding_series(symbol: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load funding rate series and resample to 1h using forward-fill.
    
    Funding rates are published every 8 hours (00:00, 08:00, 16:00 UTC).
    Forward-fill to 1h grid per DATA_PROTOCOL.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT")
    data_dir : str
        Base data directory (default: "data/raw")
        
    Returns
    -------
    pd.DataFrame
        Funding rates with column: fundingRate
        Index: UTC DatetimeIndex at 1h intervals
    """
    path = Path(data_dir) / "futures" / f"{symbol}_funding.csv.gz"
    df = load_csv_gz(path)
    
    if df is None:
        raise FileNotFoundError(f"Funding data not found: {path}")
    
    if "fundingRate" not in df.columns:
        raise ValueError(f"Missing 'fundingRate' column in {path}")
    
    # Resample to 1h with forward-fill
    df_1h = df[["fundingRate"]].resample("1h").ffill()
    
    return df_1h


def load_oi_series(symbol: str, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load open interest series and resample to 1h.
    
    Uses last observation in each 1h window per DATA_PROTOCOL.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT")
    data_dir : str
        Base data directory (default: "data/raw")
        
    Returns
    -------
    pd.DataFrame
        Open interest with column: sumOpenInterest
        Index: UTC DatetimeIndex at 1h intervals
    """
    path = Path(data_dir) / "futures" / f"{symbol}_oi.csv.gz"
    df = load_csv_gz(path)
    
    if df is None:
        raise FileNotFoundError(f"Open interest data not found: {path}")
    
    if "sumOpenInterest" not in df.columns:
        raise ValueError(f"Missing 'sumOpenInterest' column in {path}")
    
    # Resample to 1h using last observation
    df_1h = df[["sumOpenInterest"]].resample("1h").last()
    
    return df_1h


def load_basis_series(symbol: str, data_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """
    Load basis series and resample to 1h.
    
    If basis endpoint data is unavailable, compute from mark - spot.
    Uses last observation in each 1h window per DATA_PROTOCOL.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT")
    data_dir : str
        Base data directory (default: "data/raw")
        
    Returns
    -------
    pd.DataFrame or None
        Basis with column: basis
        Index: UTC DatetimeIndex at 1h intervals
        Returns None if basis data cannot be loaded
    """
    path = Path(data_dir) / "futures" / f"{symbol}_basis.csv.gz"
    df = load_csv_gz(path)
    
    if df is not None and "basis" in df.columns:
        # Resample to 1h using last observation
        df_1h = df[["basis"]].resample("1h").last()
        return df_1h
    
    # Fallback: compute from mark - spot
    try:
        mark_path = Path(data_dir) / "futures" / f"{symbol}_mark_1h.csv.gz"
        mark_df = load_csv_gz(mark_path)
        spot_df = load_spot_series(symbol, data_dir)
        
        if mark_df is not None and "close" in mark_df.columns:
            # Align indices
            common_idx = mark_df.index.intersection(spot_df.index)
            basis = mark_df.loc[common_idx, "close"] - spot_df.loc[common_idx, "close"]
            df_1h = pd.DataFrame({"basis": basis})
            return df_1h
    except Exception:
        pass
    
    return None
