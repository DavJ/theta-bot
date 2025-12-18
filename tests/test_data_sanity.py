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
    
    assert stats["appears_synthetic"] is True
    assert stats["warning_message"] is not None
    assert "unrealistic" in stats["warning_message"].lower() or "wide range" in stats["warning_message"].lower()


def test_realistic_data_no_warning():
    """Test that realistic data does not trigger warnings."""
    # Create data with realistic BTC price range for 2024
    idx = pd.date_range("2024-06-01", periods=100, freq="h")
    df = pd.DataFrame({
        "open": range(60000, 70000, 100),
        "high": range(60100, 70100, 100),
        "low": range(59900, 69900, 100),
        "close": range(60000, 70000, 100),
        "volume": [100] * 100,
    }, index=idx)
    
    stats = check_price_sanity(df, symbol="BTCUSDT")
    
    assert stats["appears_synthetic"] is False
    assert stats["warning_message"] is None


def test_check_price_sanity_returns_stats():
    """Test that check_price_sanity returns expected fields."""
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame({
        "open": [50000] * 10,
        "high": [51000] * 10,
        "low": [49000] * 10,
        "close": [50000] * 10,
        "volume": [100] * 10,
    }, index=idx)
    
    stats = check_price_sanity(df)
    
    assert "min_close" in stats
    assert "max_close" in stats
    assert "mean_close" in stats
    assert "start_timestamp" in stats
    assert "end_timestamp" in stats
    assert "num_rows" in stats
    assert "appears_synthetic" in stats
    
    assert stats["min_close"] == 50000
    assert stats["max_close"] == 50000
    assert stats["num_rows"] == 10
