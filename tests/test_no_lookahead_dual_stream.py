"""Test that dual-stream features do not leak future information (causality)."""

import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging.features import build_dual_stream_inputs


def test_no_lookahead_theta_mellin():
    """
    Verify that modifying future close values does not affect past features.
    
    This confirms causality: features at time t use only data up to time t.
    """
    np.random.seed(42)
    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    
    # Create synthetic data
    t = np.linspace(0, 4 * np.pi, n)
    prices = 100 + 5 * np.sin(t) + np.random.randn(n) * 0.3
    
    df = pd.DataFrame(
        {
            "open": prices + 0.1,
            "high": prices + 0.2,
            "low": prices - 0.1,
            "close": prices,
            "volume": 1000 + np.random.rand(n) * 50,
        },
        index=idx,
    )
    
    window = 48
    mellin_k = 16
    
    # Build inputs for original data
    index_orig, X_theta_orig, X_mellin_orig = build_dual_stream_inputs(
        df, window=window, q=0.9, n_terms=8, mellin_k=mellin_k
    )
    
    # Choose a split point (ensure enough data before and after)
    split_idx = 70
    
    # Modify future close values (after split_idx)
    df_modified = df.copy()
    df_modified.loc[df_modified.index[split_idx:], "close"] = 99999.0
    
    # Rebuild inputs with modified future data
    index_mod, X_theta_mod, X_mellin_mod = build_dual_stream_inputs(
        df_modified, window=window, q=0.9, n_terms=8, mellin_k=mellin_k
    )
    
    # Find overlapping valid indices before split_idx
    # Due to window, first valid index is at position window-1
    valid_before_split = [i for i in range(len(index_orig)) if index_orig[i] < df.index[split_idx]]
    
    if len(valid_before_split) == 0:
        pytest.skip("Not enough valid samples before split for meaningful test")
    
    # Check that features before split_idx are unchanged
    for i in valid_before_split:
        timestamp = index_orig[i]
        
        # Find corresponding index in modified data
        if timestamp not in index_mod:
            continue
        
        idx_mod = index_mod.get_loc(timestamp)
        
        # Features should be identical (or very close due to numerical precision)
        theta_diff = np.abs(X_theta_orig[i] - X_theta_mod[idx_mod])
        mellin_diff = np.abs(X_mellin_orig[i] - X_mellin_mod[idx_mod])
        
        # Allow small numerical tolerance
        assert np.allclose(X_theta_orig[i], X_theta_mod[idx_mod], atol=1e-6), \
            f"Theta features at {timestamp} changed when future data modified"
        assert np.allclose(X_mellin_orig[i], X_mellin_mod[idx_mod], atol=1e-6), \
            f"Mellin features at {timestamp} changed when future data modified"


def test_theta_coefficients_causality():
    """
    Direct test that theta coefficients at time t only depend on data up to t.
    """
    from theta_bot_averaging.features import build_theta_embedding
    
    np.random.seed(123)
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    close = pd.Series(prices, index=idx)
    
    window = 30
    n_terms = 5
    
    # Build theta embedding for original data
    result_orig = build_theta_embedding(close, window=window, q=0.9, n_terms=n_terms)
    theta_coeffs_orig = result_orig["theta_coeffs"]
    
    # Modify future values (after position 50)
    close_modified = close.copy()
    close_modified.iloc[50:] = 200.0
    
    # Rebuild theta embedding
    result_mod = build_theta_embedding(close_modified, window=window, q=0.9, n_terms=n_terms)
    theta_coeffs_mod = result_mod["theta_coeffs"]
    
    # Check that coefficients before position 50 are unchanged
    # Account for window: valid coefficients start at position window-1
    for i in range(window - 1, 50):
        # Allow for numerical tolerance
        assert np.allclose(theta_coeffs_orig[i], theta_coeffs_mod[i], atol=1e-6, equal_nan=True), \
            f"Theta coefficients at position {i} changed when future data modified"
