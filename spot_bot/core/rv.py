"""
Helper functions for realized volatility (RV) computation.

Centralizes rv_ref calculation to ensure consistency across all modes.
"""

from __future__ import annotations

import pandas as pd


def compute_rv_ref_series(rv_series: pd.Series, window: int = 500) -> pd.Series:
    """
    Compute reference realized volatility series using rolling median.
    
    This is the canonical implementation used by fast_backtest, replay, and live modes.
    
    Args:
        rv_series: Series of realized volatility values
        window: Rolling window size (default 500 bars)
    
    Returns:
        Series of rv_ref values (rolling median with forward fill)
        
    The function:
    1. Computes rolling median with min_periods=1 (works from first bar)
    2. Forward fills any NaN values
    3. Defaults to 1.0 for any remaining NaN
    """
    rv_ref = rv_series.rolling(window=window, min_periods=1).median()
    rv_ref = rv_ref.fillna(method='ffill')
    rv_ref = rv_ref.fillna(1.0)
    return rv_ref


def compute_rv_ref_scalar(rv_series: pd.Series, window: int = 500) -> float:
    """
    Compute single rv_ref value from a series.
    
    Used in live streaming where we compute rv_ref for the latest bar.
    
    Args:
        rv_series: Series of realized volatility values
        window: Rolling window size (default 500 bars)
    
    Returns:
        Single rv_ref value (median of last `window` bars)
    """
    rv_series_clean = rv_series.dropna()
    if rv_series_clean.empty:
        return 1.0
    
    # Take last `window` bars
    rv_candidates = rv_series_clean.tail(window)
    
    # Compute median
    rv_ref = float(rv_candidates.median()) if not rv_candidates.empty else 1.0
    
    # Ensure positive
    if rv_ref <= 0.0:
        # Fallback to absolute value of current or 1.0
        rv_current = float(rv_series_clean.iloc[-1]) if not rv_series_clean.empty else 0.0
        rv_ref = max(abs(rv_current), 1.0)
    
    return rv_ref


__all__ = [
    "compute_rv_ref_series",
    "compute_rv_ref_scalar",
]
