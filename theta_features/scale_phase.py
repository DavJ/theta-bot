from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_scale_phase(rv: pd.Series, window: int | None, base: float = 10.0, eps: float = 1e-12) -> pd.Series:
    """
    Compute scale-phase psi from realized volatility.

    psi_t = frac(log(RV_t / median(RV_{t-W:t})) / log(base)), psi in [0, 1)
    """
    if base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive and not equal to 1")
    if window is None or window <= 0:
        return pd.Series(np.nan, index=rv.index if isinstance(rv, pd.Series) else None)

    rv_series = pd.to_numeric(rv, errors="coerce")
    med = rv_series.rolling(window=window, min_periods=window).median()
    ratio = rv_series / med
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.where(ratio > 0.0)

    log_base = math.log(base)
    log_ratio = np.log(np.maximum(ratio, eps)) / log_base
    psi = log_ratio - np.floor(log_ratio)
    return psi
