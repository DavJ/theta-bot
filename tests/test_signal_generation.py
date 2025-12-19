"""
Tests for signal generation utilities.
"""
import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging.utils import generate_signals


def test_generate_signals_threshold_mode():
    """Test signal generation in threshold mode."""
    # Create test data
    pred_return = pd.Series([0.002, -0.002, 0.0001, -0.0001, 0.0, np.nan])
    
    # Generate signals with thresholds
    signals = generate_signals(
        pred_return,
        mode="threshold",
        positive_threshold=0.001,
        negative_threshold=-0.001,
    )
    
    # Expected signals: [1, -1, 0, 0, 0, 0]
    expected = pd.Series([1, -1, 0, 0, 0, 0], dtype=int)
    pd.testing.assert_series_equal(signals, expected)


def test_generate_signals_quantile_mode():
    """Test signal generation in quantile mode."""
    # Create test data with 100 samples
    np.random.seed(42)
    pred_return = pd.Series(np.random.randn(100))
    
    # Generate signals with quantiles
    signals = generate_signals(
        pred_return,
        mode="quantile",
        quantile_long=0.95,
        quantile_short=0.05,
    )
    
    # Check that approximately 5% are long and 5% are short
    long_count = (signals == 1).sum()
    short_count = (signals == -1).sum()
    neutral_count = (signals == 0).sum()
    
    # Should be approximately 5 long, 5 short, 90 neutral
    assert 3 <= long_count <= 7, f"Expected ~5 long signals, got {long_count}"
    assert 3 <= short_count <= 7, f"Expected ~5 short signals, got {short_count}"
    assert 86 <= neutral_count <= 94, f"Expected ~90 neutral signals, got {neutral_count}"
    
    # Check that long signals correspond to high predicted returns
    long_threshold = pred_return.quantile(0.95)
    assert all(pred_return[signals == 1] > long_threshold * 0.99)  # Allow small tolerance
    
    # Check that short signals correspond to low predicted returns
    short_threshold = pred_return.quantile(0.05)
    assert all(pred_return[signals == -1] < short_threshold * 1.01)  # Allow small tolerance


def test_generate_signals_quantile_with_nans():
    """Test quantile mode handles NaN values correctly."""
    pred_return = pd.Series([0.5, -0.5, 0.3, -0.3, np.nan, np.nan, 0.1, -0.1])
    
    signals = generate_signals(
        pred_return,
        mode="quantile",
        quantile_long=0.85,
        quantile_short=0.15,
    )
    
    # NaN values should result in neutral signals
    assert signals.iloc[4] == 0
    assert signals.iloc[5] == 0
    
    # Non-NaN values should have signals
    assert len(signals[signals != 0]) > 0


def test_generate_signals_empty_series():
    """Test handling of empty series."""
    pred_return = pd.Series([], dtype=float)
    
    signals = generate_signals(pred_return, mode="threshold")
    assert len(signals) == 0
    
    signals = generate_signals(pred_return, mode="quantile")
    assert len(signals) == 0


def test_generate_signals_all_nans():
    """Test handling of all-NaN series."""
    pred_return = pd.Series([np.nan, np.nan, np.nan])
    
    signals = generate_signals(pred_return, mode="quantile")
    assert all(signals == 0)


def test_generate_signals_invalid_mode():
    """Test error handling for invalid mode."""
    pred_return = pd.Series([0.1, -0.1])
    
    with pytest.raises(ValueError, match="Unknown signal mode"):
        generate_signals(pred_return, mode="invalid_mode")


def test_generate_signals_preserves_index():
    """Test that signal series preserves the index of input series."""
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    pred_return = pd.Series([0.002, -0.002, 0.0001, -0.0001, 0.0], index=index)
    
    signals = generate_signals(pred_return, mode="threshold")
    
    pd.testing.assert_index_equal(signals.index, pred_return.index)


def test_generate_signals_quantile_extreme():
    """Test quantile mode with extreme quantiles."""
    pred_return = pd.Series(np.random.randn(100))
    
    # Very aggressive quantiles (top 1% and bottom 1%)
    signals = generate_signals(
        pred_return,
        mode="quantile",
        quantile_long=0.99,
        quantile_short=0.01,
    )
    
    long_count = (signals == 1).sum()
    short_count = (signals == -1).sum()
    
    # Should be approximately 1 long and 1 short
    assert 0 <= long_count <= 3
    assert 0 <= short_count <= 3
