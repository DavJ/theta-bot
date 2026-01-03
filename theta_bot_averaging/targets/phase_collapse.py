"""
Phase-collapse event target constructors.

These targets define events (regime/volatility shifts) rather than directional returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_vol_burst_labels(
    close: pd.Series,
    horizon: int = 8,
    quantile: float = 0.80,
) -> tuple[pd.Series, pd.Series]:
    """
    Create VOL_BURST binary event labels based on future realized volatility.
    
    An event (label=1) occurs when future realized volatility over the next H bars
    is high (>= quantile threshold).
    
    This is a STRICTLY forward-looking target with no leakage:
    - r[t] = log(close[t]) - log(close[t-1])
    - future_vol[t] = std(r[t+1 : t+H+1])  # STRICTLY future window
    - event[t] = 1 if future_vol[t] >= threshold else 0
    - Last H rows are dropped (no future window available)
    
    **Indexing Example (horizon=3):**
    For close prices at indices [0, 1, 2, 3, 4, 5]:
    - At t=0: future_vol[0] = std([r[1], r[2], r[3]]) where r[i] = log(close[i]/close[i-1])
    - At t=1: future_vol[1] = std([r[2], r[3], r[4]])
    - At t=2: future_vol[2] = std([r[3], r[4], r[5]])
    - At t=3: Cannot compute (not enough future data) → dropped
    
    Parameters
    ----------
    close : pd.Series
        Close price series with DatetimeIndex
    horizon : int, default=8
        Number of bars to look ahead for volatility computation
    quantile : float, default=0.80
        Quantile threshold for defining high volatility events (0-1)
    
    Returns
    -------
    labels : pd.Series[int]
        Binary event labels (1=event, 0=no event), aligned with input index
        minus the last H rows
    future_vol : pd.Series[float]
        Future realized volatility values, for fold-safe threshold computation
    
    Notes
    -----
    To avoid leakage in cross-validation:
    - The quantile threshold should be computed on TRAIN data only (per fold)
    - This function returns both labels and future_vol
    - In evaluation, compute threshold = quantile(future_vol_train, q)
    - Then define event_train and event_test using the TRAIN threshold
    
    Examples
    --------
    >>> close = pd.Series([100, 102, 101, 105, 103], index=pd.date_range("2024-01-01", periods=5, freq="H"))
    >>> labels, future_vol = make_vol_burst_labels(close, horizon=2, quantile=0.80)
    >>> len(labels)  # Should be 3 (5 - 2)
    3
    """
    # Compute log returns
    log_close = np.log(close)
    returns = log_close.diff()  # r[t] = log(close[t]) - log(close[t-1])
    
    # Compute future volatility: std(r[t+1:t+H+1]) for each t
    # This is a rolling window applied to FUTURE returns
    future_vol = pd.Series(index=close.index, dtype=float)
    
    n = len(close)
    for i in range(n):
        # Future window: [i+1, i+H+1) in returns array
        # Since returns[t] = log(close[t]) - log(close[t-1]),
        # returns[i+1:i+H+1] spans from close[i+1] to close[i+H]
        future_start = i + 1
        future_end = i + horizon + 1
        
        if future_end <= n:
            # Extract future returns
            future_rets = returns.iloc[future_start:future_end]
            # Compute std (NaN if empty or all NaN)
            vol = future_rets.std()
            future_vol.iloc[i] = vol
        else:
            # Not enough future data
            future_vol.iloc[i] = np.nan
    
    # Drop rows where future_vol is NaN (last H rows)
    valid_mask = ~future_vol.isna()
    future_vol = future_vol[valid_mask]
    
    # Compute threshold from available data
    # NOTE: In cross-validation, this should be recomputed on train set only
    threshold = future_vol.quantile(quantile)
    
    # Create binary labels
    labels = (future_vol >= threshold).astype(int)
    
    return labels, future_vol
