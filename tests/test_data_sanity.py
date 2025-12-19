"""Test data sanity check utility."""

import pandas as pd
import pytest
from theta_bot_averaging.utils.data_sanity import check_price_sanity


def test_synthetic_data_detection():
    """Test that synthetic/unrealistic data is detected."""
    # Create synthetic data with unrealistic price range
    idx = pd.date_range("2024-06-01", periods=100, freq="h")
    df = pd.DataFrame({
        "open": range(10000, 110000, 1000),
        "high": range(10100, 110100, 1000),
        "low": range(9900, 109900, 1000),
        "close": range(10000, 110000, 1000),
        "volume": [100] * 100,
    }, index=idx)
    
    stats = check_price_sanity(df, symbol="BTCUSDT")
    
    assert stats["is_realistic"] is False
    assert stats["appears_unrealistic"] is True
    assert len(stats["failed_checks"]) > 0
    assert stats["warning_message"] is not None


def test_synthetic_data_strict_mode_raises():
    """Test that strict mode raises ValueError for unrealistic data."""
    # Create synthetic data with unrealistic price range
    idx = pd.date_range("2024-06-01", periods=100, freq="h")
    df = pd.DataFrame({
        "open": range(10000, 110000, 1000),
        "high": range(10100, 110100, 1000),
        "low": range(9900, 109900, 1000),
        "close": range(10000, 110000, 1000),
        "volume": [100] * 100,
    }, index=idx)
    
    with pytest.raises(ValueError, match="Dataset does not look real"):
        check_price_sanity(df, symbol="BTCUSDT", strict=True)


def test_realistic_data_no_warning():
    """Test that realistic data does not trigger warnings."""
    # Create data with realistic BTC price range for 2024
    idx = pd.date_range("2024-06-01", periods=2000, freq="h")
    df = pd.DataFrame({
        "open": [60000 + i * 5 for i in range(2000)],
        "high": [60100 + i * 5 for i in range(2000)],
        "low": [59900 + i * 5 for i in range(2000)],
        "close": [60000 + i * 5 for i in range(2000)],
        "volume": [100] * 2000,
    }, index=idx)
    
    stats = check_price_sanity(df, symbol="BTCUSDT")
    
    assert stats["is_realistic"] is True
    assert stats["appears_unrealistic"] is False
    assert stats["warning_message"] is None
    assert len(stats["failed_checks"]) == 0


def test_insufficient_rows_detected():
    """Test that insufficient row count is detected."""
    idx = pd.date_range("2024-01-01", periods=500, freq="h")
    df = pd.DataFrame({
        "open": [50000] * 500,
        "high": [51000] * 500,
        "low": [49000] * 500,
        "close": [50000] * 500,
        "volume": [100] * 500,
    }, index=idx)
    
    stats = check_price_sanity(df)
    
    assert stats["is_realistic"] is False
    assert "Insufficient data" in str(stats["failed_checks"])


def test_nan_values_detected():
    """Test that NaN values are detected."""
    idx = pd.date_range("2024-01-01", periods=2000, freq="h")
    df = pd.DataFrame({
        "open": [50000] * 2000,
        "high": [51000] * 2000,
        "low": [49000] * 2000,
        "close": [50000] * 1999 + [float('nan')],
        "volume": [100] * 2000,
    }, index=idx)
    
    stats = check_price_sanity(df)
    
    assert stats["is_realistic"] is False
    assert any("NaN" in check for check in stats["failed_checks"])


def test_non_monotonic_timestamps_detected():
    """Test that non-monotonic timestamps are detected."""
    # Create a non-monotonic timestamp sequence by manually shuffling some dates
    dates = pd.date_range("2024-01-01", periods=1500, freq="h")
    # Take first 3 dates and swap them to make non-monotonic
    idx_list = [dates[0], dates[2], dates[1]] + list(dates[3:])
    idx = pd.DatetimeIndex(idx_list)
    
    df = pd.DataFrame({
        "open": [50000] * len(idx),
        "high": [51000] * len(idx),
        "low": [49000] * len(idx),
        "close": [50000] * len(idx),
        "volume": [100] * len(idx),
    }, index=idx)
    
    stats = check_price_sanity(df)
    
    assert stats["is_realistic"] is False
    assert any("monotonic" in check.lower() for check in stats["failed_checks"])


def test_check_price_sanity_returns_stats():
    """Test that check_price_sanity returns expected fields."""
    idx = pd.date_range("2024-01-01", periods=2000, freq="h")
    df = pd.DataFrame({
        "open": [50000] * 2000,
        "high": [51000] * 2000,
        "low": [49000] * 2000,
        "close": [50000] * 2000,
        "volume": [100] * 2000,
    }, index=idx)
    
    stats = check_price_sanity(df)
    
    assert "min_close" in stats
    assert "max_close" in stats
    assert "mean_close" in stats
    assert "start_timestamp" in stats
    assert "end_timestamp" in stats
    assert "num_rows" in stats
    assert "is_realistic" in stats
    assert "appears_unrealistic" in stats
    assert "failed_checks" in stats
    
    assert stats["min_close"] == 50000
    assert stats["max_close"] == 50000
    assert stats["num_rows"] == 2000
