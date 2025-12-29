#!/usr/bin/env python3
"""
Feature engineering for derivatives state module.

Compute z-scores, rolling statistics, and derived features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_zscore(
    series: pd.Series,
    window: str = "7D",
    min_periods: int = 24,
) -> pd.Series:
    """
    Compute z-score normalization with rolling window.
    
    z(x) = (x - rolling_mean) / rolling_std
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    window : str
        Rolling window size (default: "7D" for 7 days)
    min_periods : int
        Minimum number of observations required (default: 24 for 1 day)
        
    Returns
    -------
    pd.Series
        Z-score normalized series
    """
    # Compute rolling statistics
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    
    # Compute z-score, avoiding division by zero
    # Use small epsilon instead of NaN to prevent NaN propagation
    zscore = (series - rolling_mean) / rolling_std.replace(0, 1e-10)
    
    return zscore


def compute_oi_change(oi_series: pd.Series) -> pd.Series:
    """
    Compute OI change as log-difference.
    
    OI'(t) = diff(log(OI(t)))
    
    Parameters
    ----------
    oi_series : pd.Series
        Open interest time series
        
    Returns
    -------
    pd.Series
        Log-difference of open interest
    """
    # Compute log-difference
    log_oi = np.log(oi_series.replace(0, np.nan))
    oi_change = log_oi.diff()
    
    return oi_change


def compute_spot_return(spot_series: pd.Series) -> pd.Series:
    """
    Compute spot return as log-difference.
    
    r(t) = log(close(t)) - log(close(t-1))
    
    Parameters
    ----------
    spot_series : pd.Series
        Spot price close series
        
    Returns
    -------
    pd.Series
        Log-return of spot price
    """
    log_price = np.log(spot_series.replace(0, np.nan))
    returns = log_price.diff()
    
    return returns


def align_series(*series_list: pd.Series, method: str = "inner") -> list[pd.Series]:
    """
    Align multiple time series to common index.
    
    Parameters
    ----------
    *series_list : pd.Series
        Variable number of Series to align
    method : str
        Join method: "inner" (intersection) or "outer" (union)
        
    Returns
    -------
    list[pd.Series]
        List of aligned series
    """
    if not series_list:
        return []
    
    # Get common index
    if method == "inner":
        common_idx = series_list[0].index
        for s in series_list[1:]:
            common_idx = common_idx.intersection(s.index)
    else:  # outer
        common_idx = series_list[0].index
        for s in series_list[1:]:
            common_idx = common_idx.union(s.index)
        common_idx = common_idx.sort_values()
    
    # Reindex all series
    aligned = [s.reindex(common_idx) for s in series_list]
    
    return aligned
