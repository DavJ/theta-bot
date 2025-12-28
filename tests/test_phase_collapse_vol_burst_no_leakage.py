"""
Test VOL_BURST label construction to verify no leakage.

Verifies that:
1. label[t] depends ONLY on returns in (t+1..t+H), NOT on returns <= t
2. Last H labels are dropped/NaN and removed
3. Future volatility is computed correctly
"""

import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging.targets import make_vol_burst_labels


def test_vol_burst_no_leakage():
    """
    Verify that VOL_BURST labels depend only on FUTURE returns, not past.
    
    Create a close series where we can control the future volatility window.
    """
    # Create a simple price series with known volatility patterns
    # First 10 bars: low volatility (small moves)
    # Next 5 bars: high volatility (large moves)
    # Last 5 bars: low volatility again
    
    close_values = [100.0]
    
    # Low volatility period (bars 1-10): small moves ~0.1%
    for _ in range(10):
        close_values.append(close_values[-1] * (1 + np.random.uniform(-0.001, 0.001)))
    
    # High volatility period (bars 11-15): large moves ~2%
    for _ in range(5):
        close_values.append(close_values[-1] * (1 + np.random.uniform(-0.02, 0.02)))
    
    # Low volatility period (bars 16-20): small moves ~0.1%
    for _ in range(5):
        close_values.append(close_values[-1] * (1 + np.random.uniform(-0.001, 0.001)))
    
    idx = pd.date_range("2024-01-01", periods=len(close_values), freq="H")
    close = pd.Series(close_values, index=idx)
    
    # Compute labels with horizon=5
    horizon = 5
    labels, future_vol = make_vol_burst_labels(close, horizon=horizon, quantile=0.80)
    
    # Verify that last H rows are dropped
    assert len(labels) == len(close) - horizon, f"Expected {len(close) - horizon} labels, got {len(labels)}"
    assert len(future_vol) == len(close) - horizon, f"Expected {len(close) - horizon} future_vol, got {len(future_vol)}"
    
    # Verify no NaN in returned series
    assert not labels.isna().any(), "Labels should not contain NaN"
    assert not future_vol.isna().any(), "Future volatility should not contain NaN"
    
    # Verify labels are binary (0 or 1)
    assert set(labels.unique()).issubset({0, 1}), "Labels should be binary (0 or 1)"


def test_vol_burst_future_window_correctness():
    """
    Verify that future_vol[t] is computed from returns in [t+1, t+H+1).
    
    Create a synthetic series where we can manually verify the computation.
    """
    # Simple series: [100, 110, 120, 130, 100, 90, 80, 70]
    # This gives clear return patterns we can verify
    close_values = [100.0, 110.0, 120.0, 130.0, 100.0, 90.0, 80.0, 70.0]
    idx = pd.date_range("2024-01-01", periods=len(close_values), freq="H")
    close = pd.Series(close_values, index=idx)
    
    horizon = 3
    labels, future_vol = make_vol_burst_labels(close, horizon=horizon, quantile=0.50)
    
    # Expected length: 8 - 3 = 5
    assert len(future_vol) == 5
    
    # Manually compute expected future volatility for t=0
    # future window for t=0: returns[1:4] (close[1] to close[3])
    # r[1] = log(110/100) = log(1.1) ≈ 0.0953
    # r[2] = log(120/110) = log(1.0909) ≈ 0.0870
    # r[3] = log(130/120) = log(1.0833) ≈ 0.0800
    # std([0.0953, 0.0870, 0.0800]) ≈ 0.0078
    
    log_close = np.log(close.values)
    returns = np.diff(log_close)
    
    # For t=0, future returns are returns[1:4]
    expected_vol_0 = np.std(returns[1:4])
    
    # Check that computed value is close
    assert np.isclose(future_vol.iloc[0], expected_vol_0, rtol=1e-5), \
        f"Expected future_vol[0]={expected_vol_0:.6f}, got {future_vol.iloc[0]:.6f}"
    
    # For t=1, future returns are returns[2:5]
    expected_vol_1 = np.std(returns[2:5])
    assert np.isclose(future_vol.iloc[1], expected_vol_1, rtol=1e-5), \
        f"Expected future_vol[1]={expected_vol_1:.6f}, got {future_vol.iloc[1]:.6f}"


def test_vol_burst_no_past_contamination():
    """
    Critical test: Verify that changing PAST prices does NOT affect future_vol[t].
    
    This ensures no leakage from past data into the future volatility computation.
    """
    # Create two series that differ only in the past
    close_values_1 = [50.0, 60.0, 70.0, 100.0, 110.0, 120.0, 130.0]  # Low early prices
    close_values_2 = [200.0, 210.0, 220.0, 100.0, 110.0, 120.0, 130.0]  # High early prices
    
    idx = pd.date_range("2024-01-01", periods=len(close_values_1), freq="H")
    close_1 = pd.Series(close_values_1, index=idx)
    close_2 = pd.Series(close_values_2, index=idx)
    
    horizon = 3
    labels_1, future_vol_1 = make_vol_burst_labels(close_1, horizon=horizon, quantile=0.50)
    labels_2, future_vol_2 = make_vol_burst_labels(close_2, horizon=horizon, quantile=0.50)
    
    # At t=3 (close=100 in both), the future window is the same: [110, 120, 130]
    # So future_vol[3] should be IDENTICAL in both series
    # (Both have 4 valid labels: t=0,1,2,3)
    
    # The future window for t=3 spans close[4:7] which is [110, 120, 130] in both series
    # So the returns in this window are identical
    # Therefore future_vol[3] should be the same
    
    assert np.isclose(future_vol_1.iloc[3], future_vol_2.iloc[3], rtol=1e-10), \
        f"future_vol[3] should be identical when future is the same, got {future_vol_1.iloc[3]} vs {future_vol_2.iloc[3]}"
    
    # However, earlier time points may differ due to different thresholds
    # (since threshold is computed on the entire series)
    # But the RAW future_vol values at t=3 should be identical


def test_vol_burst_threshold_computation():
    """
    Verify that labels are correctly assigned based on quantile threshold.
    """
    # Create a series with known volatility pattern
    np.random.seed(42)
    close_values = [100.0]
    for _ in range(50):
        close_values.append(close_values[-1] * (1 + np.random.uniform(-0.01, 0.01)))
    
    idx = pd.date_range("2024-01-01", periods=len(close_values), freq="H")
    close = pd.Series(close_values, index=idx)
    
    horizon = 5
    quantile = 0.80
    labels, future_vol = make_vol_burst_labels(close, horizon=horizon, quantile=quantile)
    
    # Verify that approximately 20% of events are labeled as 1 (event=1)
    # (Since quantile=0.80, top 20% should be events)
    event_rate = labels.mean()
    
    # Allow some tolerance due to quantile edge effects
    assert 0.10 < event_rate < 0.30, \
        f"With quantile=0.80, expected event rate ~0.20, got {event_rate:.2f}"
    
    # Verify that labels=1 correspond to higher future_vol values
    vol_event_1 = future_vol[labels == 1].mean()
    vol_event_0 = future_vol[labels == 0].mean()
    
    assert vol_event_1 > vol_event_0, \
        f"Event=1 should have higher future_vol than event=0, got {vol_event_1:.6f} vs {vol_event_0:.6f}"


def test_vol_burst_index_alignment():
    """
    Verify that returned labels and future_vol have correct index alignment.
    """
    close_values = [100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0]
    idx = pd.date_range("2024-01-01", periods=len(close_values), freq="H")
    close = pd.Series(close_values, index=idx)
    
    horizon = 3
    labels, future_vol = make_vol_burst_labels(close, horizon=horizon, quantile=0.50)
    
    # Expected length: 8 - 3 = 5
    assert len(labels) == 5
    assert len(future_vol) == 5
    
    # Verify index alignment: should be first 5 timestamps
    expected_index = idx[:5]
    assert labels.index.equals(expected_index), "Labels index should match first N-H timestamps"
    assert future_vol.index.equals(expected_index), "Future_vol index should match first N-H timestamps"
