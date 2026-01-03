"""
Unit tests for target label construction and future return computation.

These tests verify that:
1. future_return[t] = close[t+horizon] / close[t] - 1
2. y[t] = +1 if future_return[t] > +threshold
         -1 if future_return[t] < -threshold
          0 otherwise
"""

import pandas as pd
import pytest

from theta_bot_averaging.data import build_targets


def test_future_return_computation():
    """Verify future_return is computed correctly: (close[t+horizon]/close[t]) - 1"""
    idx = pd.date_range("2024-01-01", periods=5, freq="H")
    close = [100.0, 110.0, 105.0, 115.0, 120.0]
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    
    # Test with horizon=1
    out = build_targets(df, horizon=1, threshold_bps=0)
    
    # For t=0: future_return = (close[1] / close[0]) - 1 = (110 / 100) - 1 = 0.1
    assert out.iloc[0]["future_return"] == pytest.approx(0.1)
    
    # For t=1: future_return = (close[2] / close[1]) - 1 = (105 / 110) - 1 = -0.04545...
    assert out.iloc[1]["future_return"] == pytest.approx(-0.045454545, rel=1e-6)
    
    # For t=2: future_return = (close[3] / close[2]) - 1 = (115 / 105) - 1 = 0.095238...
    assert out.iloc[2]["future_return"] == pytest.approx(0.095238095, rel=1e-6)
    
    # For t=3: future_return = (close[4] / close[3]) - 1 = (120 / 115) - 1 = 0.043478...
    assert out.iloc[3]["future_return"] == pytest.approx(0.043478261, rel=1e-6)
    
    # Last row should be dropped (no future data available)
    assert len(out) == 4


def test_label_sign_correctness():
    """Verify that labels have correct signs relative to future returns"""
    idx = pd.date_range("2024-01-01", periods=6, freq="H")
    
    # Construct prices to create specific future returns
    # We want: large positive return, small positive, zero, small negative, large negative
    close = [100.0, 120.0, 101.0, 100.0, 80.0, 100.0]  # Will test returns at t=0-4
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    
    # Use threshold of 100 bps = 0.01
    threshold_bps = 100.0
    out = build_targets(df, horizon=1, threshold_bps=threshold_bps)
    
    # t=0: future_return = (120/100) - 1 = 0.20 (>0.01) -> label should be +1
    assert out.iloc[0]["future_return"] == pytest.approx(0.20)
    assert out.iloc[0]["label"] == 1, "Large positive return should have label=+1"
    
    # t=1: future_return = (101/120) - 1 = -0.1583 (<-0.01) -> label should be -1
    assert out.iloc[1]["future_return"] == pytest.approx(-0.158333333, rel=1e-6)
    assert out.iloc[1]["label"] == -1, "Large negative return should have label=-1"
    
    # t=2: future_return = (100/101) - 1 = -0.0099 (>-0.01, <0.01) -> label should be 0
    assert out.iloc[2]["future_return"] == pytest.approx(-0.009900990, rel=1e-6)
    assert out.iloc[2]["label"] == 0, "Small return within threshold should have label=0"
    
    # t=3: future_return = (80/100) - 1 = -0.20 (<-0.01) -> label should be -1
    assert out.iloc[3]["future_return"] == pytest.approx(-0.20)
    assert out.iloc[3]["label"] == -1, "Large negative return should have label=-1"
    
    # t=4: future_return = (100/80) - 1 = 0.25 (>0.01) -> label should be +1
    assert out.iloc[4]["future_return"] == pytest.approx(0.25)
    assert out.iloc[4]["label"] == 1, "Large positive return should have label=+1"


def test_label_threshold_boundaries():
    """Test label assignment at exact threshold boundaries"""
    idx = pd.date_range("2024-01-01", periods=7, freq="H")
    
    # Prices chosen to create returns exactly at and near thresholds
    # threshold = 100 bps = 0.01
    close = [
        100.0,  # t=0
        101.0,  # t=1: ret = 0.01 (exactly at positive threshold)
        99.0,   # t=2: ret = -0.0198... (below negative threshold)
        100.0,  # t=3: ret = 0.0101... (above positive threshold)
        99.0,   # t=4: ret = -0.01 (exactly at negative threshold)
        99.5,   # t=5: ret = 0.00505... (within threshold)
        100.0,  # t=6
    ]
    
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    
    threshold_bps = 100.0
    out = build_targets(df, horizon=1, threshold_bps=threshold_bps)
    
    # t=0: future_return = (101/100) - 1 = 0.01 (exactly at threshold)
    # Should be labeled as +1 (> threshold)
    assert out.iloc[0]["future_return"] == pytest.approx(0.01)
    assert out.iloc[0]["label"] == 1, "Return exactly at positive threshold should have label=+1"
    
    # t=1: future_return = (99/101) - 1 = -0.0198...
    # Should be labeled as -1 (< -threshold)
    assert out.iloc[1]["future_return"] == pytest.approx(-0.019801980, rel=1e-6)
    assert out.iloc[1]["label"] == -1, "Return below negative threshold should have label=-1"
    
    # t=2: future_return = (100/99) - 1 = 0.0101...
    # Should be labeled as +1 (> threshold)
    assert out.iloc[2]["future_return"] == pytest.approx(0.010101010, rel=1e-6)
    assert out.iloc[2]["label"] == 1, "Return above positive threshold should have label=+1"
    
    # t=3: future_return = (99/100) - 1 = -0.01 (at/near negative threshold)
    # Due to floating point, this will be slightly less than -0.01
    # Should be labeled as -1
    assert out.iloc[3]["future_return"] == pytest.approx(-0.01)
    assert out.iloc[3]["label"] == -1, "Return at/below negative threshold should have label=-1"
    
    # t=4: future_return = (99.5/99) - 1 = 0.00505...
    # Should be labeled as 0 (within threshold)
    assert out.iloc[4]["future_return"] == pytest.approx(0.005050505, rel=1e-6)
    assert out.iloc[4]["label"] == 0, "Small return within threshold should have label=0"


def test_label_inversion_check():
    """
    Critical test: Verify that positive future returns get positive labels
    and negative future returns get negative labels (no inversion).
    """
    idx = pd.date_range("2024-01-01", periods=4, freq="H")
    
    # Clear cases: large moves
    close = [100.0, 150.0, 100.0, 50.0]
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    
    # Use 1000 bps = 10% threshold
    out = build_targets(df, horizon=1, threshold_bps=1000.0)
    
    # t=0: future_return = (150/100) - 1 = 0.5 (50% gain)
    # This is POSITIVE, so label must be POSITIVE
    assert out.iloc[0]["future_return"] > 0, "Return should be positive"
    assert out.iloc[0]["label"] == 1, "Positive future return MUST have label=+1"
    
    # t=1: future_return = (100/150) - 1 = -0.333... (33% loss)
    # This is NEGATIVE, so label must be NEGATIVE
    assert out.iloc[1]["future_return"] < 0, "Return should be negative"
    assert out.iloc[1]["label"] == -1, "Negative future return MUST have label=-1"
    
    # t=2: future_return = (50/100) - 1 = -0.5 (50% loss)
    # This is NEGATIVE, so label must be NEGATIVE
    assert out.iloc[2]["future_return"] < 0, "Return should be negative"
    assert out.iloc[2]["label"] == -1, "Negative future return MUST have label=-1"


def test_class_mean_returns_ordering():
    """
    Verify that when we compute class mean returns from labels,
    we get: mu[-1] < mu[0] < mu[+1]
    
    This is a fundamental sanity check: if label=-1 corresponds to
    negative future returns, then the mean return for that class
    should be negative.
    """
    idx = pd.date_range("2024-01-01", periods=10, freq="H")
    
    # Create data with clear up/down moves
    close = [100.0, 115.0, 110.0, 125.0, 120.0, 135.0, 130.0, 85.0, 90.0, 80.0]
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    
    # Use 100 bps threshold
    out = build_targets(df, horizon=1, threshold_bps=100.0)
    
    # Compute class mean returns
    class_means = out.groupby("label")["future_return"].mean()
    
    # Verify ordering: mu[-1] < mu[0] < mu[+1]
    if -1 in class_means.index and 0 in class_means.index:
        assert class_means[-1] < class_means[0], \
            f"mu[-1]={class_means[-1]} should be < mu[0]={class_means[0]}"
    
    if 0 in class_means.index and 1 in class_means.index:
        assert class_means[0] < class_means[1], \
            f"mu[0]={class_means[0]} should be < mu[+1]={class_means[1]}"
    
    if -1 in class_means.index and 1 in class_means.index:
        assert class_means[-1] < class_means[1], \
            f"mu[-1]={class_means[-1]} should be < mu[+1]={class_means[1]}"
    
    # Also verify that mu[-1] is negative and mu[+1] is positive
    if -1 in class_means.index:
        assert class_means[-1] < 0, \
            f"mu[-1]={class_means[-1]} should be negative (class -1 corresponds to negative returns)"
    
    if 1 in class_means.index:
        assert class_means[1] > 0, \
            f"mu[+1]={class_means[1]} should be positive (class +1 corresponds to positive returns)"
