#!/usr/bin/env python3
"""
Unit tests for derivatives_state module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging.derivatives_state import (
    compute_zscore,
    compute_oi_change,
    compute_drift,
    compute_determinism,
    apply_quantile_gate,
    apply_threshold_gate,
    apply_combined_gate,
)


def test_compute_zscore():
    """Test z-score computation."""
    # Create a simple time series
    dates = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    values = np.random.randn(200) * 10 + 50
    series = pd.Series(values, index=dates)
    
    # Compute z-score with 7-day window
    z = compute_zscore(series, window="7D", min_periods=24)
    
    # Check that z-score has same length
    assert len(z) == len(series)
    
    # Check that z-score has mean close to 0 (after warm-up period)
    z_valid = z.dropna()
    assert abs(z_valid.mean()) < 0.5, "Z-score mean should be close to 0"
    
    # Check that z-score has std close to 1 (after warm-up period)
    assert abs(z_valid.std() - 1.0) < 0.3, "Z-score std should be close to 1"


def test_compute_oi_change():
    """Test OI change computation."""
    # Create OI series with exponential growth
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    oi_values = 1000 * np.exp(np.linspace(0, 1, 100))
    oi_series = pd.Series(oi_values, index=dates)
    
    # Compute OI change
    oi_change = compute_oi_change(oi_series)
    
    # Check length (one less due to diff)
    assert len(oi_change) == len(oi_series)
    
    # Check that first value is NaN
    assert pd.isna(oi_change.iloc[0])
    
    # Check that changes are positive for growing series
    assert (oi_change[1:] > 0).all(), "OI change should be positive for growing series"


def test_compute_drift():
    """Test drift computation."""
    # Create sample z-scored series
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    z_oi_change = pd.Series(np.random.randn(100), index=dates)
    z_funding = pd.Series(np.random.randn(100), index=dates)
    z_basis = pd.Series(np.random.randn(100), index=dates)
    
    # Compute drift with default weights
    mu, mu1, mu2, mu3 = compute_drift(
        z_oi_change=z_oi_change,
        z_funding=z_funding,
        z_basis=z_basis,
        alpha=1.0,
        beta=1.0,
        gamma=0.0,
    )
    
    # Check lengths
    assert len(mu) == 100
    assert len(mu1) == 100
    assert len(mu2) == 100
    assert len(mu3) == 100
    
    # Check that mu = mu1 + mu2 + mu3
    np.testing.assert_array_almost_equal(mu.values, (mu1 + mu2 + mu3).values)
    
    # Check mu1 formula: -alpha * z_oi_change * z_funding
    expected_mu1 = -1.0 * z_oi_change * z_funding
    np.testing.assert_array_almost_equal(mu1.values, expected_mu1.values)
    
    # Check mu2 formula: beta * z_oi_change * z_basis
    expected_mu2 = 1.0 * z_oi_change * z_basis
    np.testing.assert_array_almost_equal(mu2.values, expected_mu2.values)
    
    # Check mu3 is zero when gamma=0
    assert (mu3 == 0).all()


def test_compute_determinism():
    """Test determinism computation."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    mu = pd.Series(np.random.randn(100) * 5, index=dates)
    
    D = compute_determinism(mu)
    
    # Check length
    assert len(D) == len(mu)
    
    # Check that D = |mu|
    np.testing.assert_array_almost_equal(D.values, np.abs(mu.values))
    
    # Check that D is always non-negative
    assert (D >= 0).all()


def test_apply_quantile_gate():
    """Test quantile-based gating."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    D = pd.Series(np.random.exponential(1.0, 100), index=dates)
    
    # Apply 85th percentile gate
    active = apply_quantile_gate(D, quantile=0.85)
    
    # Check length
    assert len(active) == len(D)
    
    # Check that approximately 15% are active
    active_pct = active.sum() / len(active)
    assert 0.10 < active_pct < 0.20, f"Expected ~15% active, got {active_pct*100:.1f}%"
    
    # Check that all active values are above or equal to threshold
    threshold = D.quantile(0.85)
    assert (D[active] >= threshold).all() or active.sum() == 0


def test_apply_threshold_gate():
    """Test fixed threshold gating."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    D = pd.Series(np.linspace(0, 10, 100), index=dates)
    
    # Apply fixed threshold
    active = apply_threshold_gate(D, threshold=5.0)
    
    # Check length
    assert len(active) == len(D)
    
    # Check that values above threshold are active
    assert (active == (D > 5.0)).all()


def test_apply_combined_gate():
    """Test combined quantile OR threshold gating."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    D = pd.Series(np.random.exponential(1.0, 100), index=dates)
    
    # Apply combined gate
    active = apply_combined_gate(D, quantile=0.85, threshold=2.0)
    
    # Check length
    assert len(active) == len(D)
    
    # Check that it's OR logic: quantile OR threshold
    active_q = apply_quantile_gate(D, quantile=0.85)
    active_t = apply_threshold_gate(D, threshold=2.0)
    expected = active_q | active_t
    
    assert (active == expected).all()


def test_drift_with_custom_weights():
    """Test drift computation with custom weights."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    z_oi_change = pd.Series(np.ones(100), index=dates)
    z_funding = pd.Series(np.ones(100), index=dates)
    z_basis = pd.Series(np.ones(100), index=dates)
    
    # Compute drift with custom weights
    mu, mu1, mu2, mu3 = compute_drift(
        z_oi_change=z_oi_change,
        z_funding=z_funding,
        z_basis=z_basis,
        alpha=2.0,
        beta=3.0,
        gamma=0.0,
    )
    
    # Check mu1: -alpha * 1 * 1 = -2.0
    assert (mu1 == -2.0).all()
    
    # Check mu2: beta * 1 * 1 = 3.0
    assert (mu2 == 3.0).all()
    
    # Check mu: -2.0 + 3.0 = 1.0
    assert (mu == 1.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
