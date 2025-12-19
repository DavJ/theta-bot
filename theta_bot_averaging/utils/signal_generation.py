"""
Signal generation utilities supporting threshold and quantile modes.
"""
from typing import Literal

import numpy as np
import pandas as pd


SignalMode = Literal["threshold", "quantile"]


def generate_signals(
    predicted_return: pd.Series,
    mode: SignalMode = "threshold",
    positive_threshold: float = 0.0005,
    negative_threshold: float = -0.0005,
    quantile_long: float = 0.95,
    quantile_short: float = 0.05,
) -> pd.Series:
    """
    Generate trading signals from predicted returns.

    Parameters
    ----------
    predicted_return : pd.Series
        Predicted return values
    mode : SignalMode
        Signal generation mode:
        - "threshold": Fixed threshold mode (default)
        - "quantile": Quantile-based mode
    positive_threshold : float
        Threshold for long signal in threshold mode (default: 0.0005)
    negative_threshold : float
        Threshold for short signal in threshold mode (default: -0.0005)
    quantile_long : float
        Quantile for long signal in quantile mode (default: 0.95 = 95th percentile)
    quantile_short : float
        Quantile for short signal in quantile mode (default: 0.05 = 5th percentile)

    Returns
    -------
    pd.Series
        Trading signals: 1 (long), -1 (short), 0 (neutral)

    Examples
    --------
    Threshold mode (default):
    >>> pred = pd.Series([0.001, -0.001, 0.0001, -0.0001])
    >>> generate_signals(pred, mode="threshold", positive_threshold=0.0005, negative_threshold=-0.0005)
    0    1
    1   -1
    2    0
    3    0
    dtype: int64

    Quantile mode:
    >>> pred = pd.Series(np.random.randn(100))
    >>> signals = generate_signals(pred, mode="quantile", quantile_long=0.95, quantile_short=0.05)
    >>> (signals == 1).sum()  # Should be ~5% of samples
    5
    """
    signal = pd.Series(0, index=predicted_return.index, dtype=int)

    if mode == "threshold":
        # Fixed threshold mode
        signal[predicted_return > positive_threshold] = 1
        signal[predicted_return < negative_threshold] = -1

    elif mode == "quantile":
        # Quantile mode: compute thresholds from data distribution
        if len(predicted_return) == 0 or predicted_return.isna().all():
            return signal

        # Remove NaN values for quantile computation
        pred_clean = predicted_return.dropna()
        if len(pred_clean) == 0:
            return signal

        # Compute quantile thresholds
        long_threshold = pred_clean.quantile(quantile_long)
        short_threshold = pred_clean.quantile(quantile_short)

        # Generate signals
        signal[predicted_return > long_threshold] = 1
        signal[predicted_return < short_threshold] = -1

    else:
        raise ValueError(f"Unknown signal mode: {mode}. Must be 'threshold' or 'quantile'.")

    return signal
