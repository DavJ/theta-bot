#!/usr/bin/env python3
"""State construction for derivatives SDE decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import loaders

EPSILON = 1e-12


def _compute_zscore(series: pd.Series, window: int, clip: float = 10.0) -> pd.Series:
    """Rolling z-score using past-only window."""
    if window <= 1:
        return series * 0.0
    rolling_mean = series.rolling(window=window, min_periods=max(1, window // 4)).mean()
    rolling_std = series.rolling(window=window, min_periods=max(1, window // 4)).std()
    z = (series - rolling_mean) / rolling_std.replace(0, EPSILON)
    if clip:
        z = z.clip(lower=-clip, upper=clip)
    return z


def _log_diff(series: pd.Series) -> pd.Series:
    log_v = np.log(series.replace(0, np.nan))
    return log_v.diff()


def build_state_from_frames(
    spot: pd.DataFrame,
    funding: pd.DataFrame,
    oi: pd.DataFrame,
    basis: pd.DataFrame | None = None,
    z_window: int = 168,
    clip: float = 10.0,
    align: str = "inner",
) -> pd.DataFrame:
    """Construct derivatives state from provided frames (useful for tests)."""
    r = _log_diff(spot["close"])
    oi_change = _log_diff(oi["sumOpenInterest"])

    series_to_align = [r, funding["fundingRate"], oi["sumOpenInterest"], oi_change]
    if basis is not None:
        series_to_align.append(basis["basis"])
    idx = loaders.align_indices(series_to_align, how=align)

    r = r.reindex(idx)
    funding_series = funding["fundingRate"].reindex(idx)
    oi_series = oi["sumOpenInterest"].reindex(idx)
    oi_change = oi_change.reindex(idx)
    basis_series = basis["basis"].reindex(idx) if basis is not None else pd.Series(index=idx, dtype=float)

    z_funding = _compute_zscore(funding_series, window=z_window, clip=clip)
    z_oi_change = _compute_zscore(oi_change, window=z_window, clip=clip)
    z_basis = _compute_zscore(basis_series, window=z_window, clip=clip)

    df = pd.DataFrame(
        {
            "r": r,
            "fundingRate": funding_series,
            "sumOpenInterest": oi_series,
            "oi_change": oi_change,
            "basis": basis_series,
            "z_funding": z_funding,
            "z_oi_change": z_oi_change,
            "z_basis": z_basis,
        },
        index=idx,
    )

    for col in ["r", "fundingRate", "sumOpenInterest", "oi_change", "basis"]:
        df[f"mask_{col}"] = df[col].notna()

    return df


def build_state(
    symbol: str,
    data_dir: str = "data/raw",
    z_window: int = 168,
    clip: float = 10.0,
    align: str = "inner",
) -> pd.DataFrame:
    """Load raw series and assemble standardized derivatives state."""
    spot = loaders.load_spot(symbol, data_dir=data_dir)
    funding = loaders.load_funding(symbol, data_dir=data_dir)
    oi = loaders.load_oi(symbol, data_dir=data_dir)
    basis = loaders.load_basis(symbol, data_dir=data_dir)
    return build_state_from_frames(
        spot=spot,
        funding=funding,
        oi=oi,
        basis=basis,
        z_window=z_window,
        clip=clip,
        align=align,
    )
