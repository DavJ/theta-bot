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
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    _assert_schema(df)
    return df


@dataclass
class DatasetBundle:
    df: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
