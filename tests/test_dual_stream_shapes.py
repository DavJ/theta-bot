"""Test dual-stream feature extraction shapes and validity."""

import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging.features import build_dual_stream_inputs


def test_dual_stream_shapes():
    """Test that dual-stream inputs have correct shapes and no NaNs."""
    # Create synthetic OHLCV data
    np.random.seed(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    # Monotonic close with some noise
    t = np.linspace(0, 4 * np.pi, n)
    prices = 100 + 10 * np.sin(t) + np.random.randn(n) * 0.5

    df = pd.DataFrame(
        {
            "open": prices + 0.1,
            "high": prices + 0.3,
            "low": prices - 0.2,
            "close": prices,
            "volume": 1000 + np.random.rand(n) * 100,
        },
        index=idx,
    )

    # Build dual-stream inputs
    window = 48
    mellin_k = 16

    index, X_theta, X_mellin = build_dual_stream_inputs(
        df, window=window, q=0.9, n_terms=8, mellin_k=mellin_k
    )

    # Check types
    assert isinstance(index, pd.DatetimeIndex)
    assert isinstance(X_theta, np.ndarray)
    assert isinstance(X_mellin, np.ndarray)

    # Check shapes
    N = len(index)
    assert X_theta.shape == (N, window)
    assert X_mellin.shape == (N, mellin_k)

    # Check no NaNs
    assert not np.isnan(X_theta).any()
    assert not np.isnan(X_mellin).any()

    # Check index alignment
    assert len(index) == X_theta.shape[0]
    assert len(index) == X_mellin.shape[0]

    # Check that some rows were dropped (due to initial window)
    assert len(index) < len(df)
    assert len(index) >= len(df) - window + 1


def test_dual_stream_small_window():
    """Test with smaller window to verify causality."""
    np.random.seed(123)
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.1,
            "low": prices - 0.1,
            "close": prices,
            "volume": 1000,
        },
        index=idx,
    )

    window = 20
    mellin_k = 8

    index, X_theta, X_mellin = build_dual_stream_inputs(
        df, window=window, q=0.85, n_terms=5, mellin_k=mellin_k
    )

    # Verify dimensions
    N = len(index)
    assert X_theta.shape[0] == N
    assert X_theta.shape[1] == window
    assert X_mellin.shape == (N, mellin_k)

    # No NaNs
    assert not np.isnan(X_theta).any()
    assert not np.isnan(X_mellin).any()
