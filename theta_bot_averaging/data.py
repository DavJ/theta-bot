from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["vwap", "trades"]


class SchemaError(ValueError):
    pass


def _assert_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaError("Index must be a DatetimeIndex representing UTC close times.")
    if not df.index.is_monotonic_increasing:
        raise SchemaError("Index must be sorted in increasing order.")
    if df.index.duplicated().any():
        raise SchemaError("Index contains duplicate timestamps.")


def compute_future_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Strict forward-shifted future return using close price.
    """
    future = df["close"].shift(-horizon)
    return (future / df["close"]) - 1.0


def build_targets(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold_bps: float = 10.0,
    label_zero: bool = True,
) -> pd.DataFrame:
    """
    Add future_return and discrete label columns.
    """
    _assert_schema(df)
    out = df.copy()
    out["future_return"] = compute_future_return(out, horizon)
    thr = threshold_bps / 10_000.0
    labels = pd.Series(0, index=out.index, dtype=int)
    labels[out["future_return"] > thr] = 1
    labels[out["future_return"] < -thr] = -1
    if not label_zero:
        labels = labels.replace(0, pd.NA)
    out["label"] = labels
    # Drop rows with NaN targets (tail where future_return is NaN)
    out = out.dropna(subset=["future_return"])
    return out


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load OHLCV dataset from CSV with support for Binance kline format.
    
    Handles:
    - Millisecond epoch timestamps (Binance format)
    - String datetime formats
    - Converts Binance openTime to closeTime for 1h candles (+1h shift)
    
    Parameters
    ----------
    path : str
        Path to CSV file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex (UTC close times) and OHLCV columns
    """
    # Read CSV without parse_dates to avoid warnings on numeric timestamps
    df = pd.read_csv(path, index_col=0)
    
    # Determine if we need to parse timestamps
    index_col_name = df.index.name if df.index.name else "index"
    is_numeric_index = pd.api.types.is_numeric_dtype(df.index)
    
    # Check if this looks like millisecond epoch timestamps
    # Binance uses ms timestamps >= 10^12 (e.g., 1711929600000)
    is_ms_epoch = False
    if is_numeric_index:
        # Check if values are in the ms epoch range (>= 10^12)
        if len(df.index) > 0 and df.index[0] >= 10**12:
            is_ms_epoch = True
    
    # Convert timestamps based on format
    if index_col_name == "timestamp" or is_ms_epoch:
        # Binance kline format: timestamp column with ms epoch values
        # These represent candle OPEN times
        df.index = pd.to_datetime(df.index.astype("int64"), unit="ms", utc=True)
        
        # For 1h candles, shift by +1h to get CLOSE times
        # Binance provides openTime, but we want closeTime for predictions
        # (A 1h candle opened at 00:00 closes at 01:00)
        df.index = df.index + pd.Timedelta(hours=1)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try parsing as string datetime
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        # Already DatetimeIndex but no timezone - localize to UTC
        df.index = df.index.tz_localize("UTC")
    
    _assert_schema(df)
    return df


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
