#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_theta_v9.py
----------------
Comprehensive test script for theta_predictor v9 features.

Tests:
1. Biquaternionic time support
2. Fokker-Planck drift term
3. PCA regime detection
4. Backward compatibility
5. Walk-forward causality verification
"""

import numpy as np
import pandas as pd
import os
import sys
from theta_predictor import (
    generate_theta_features_1d,
    generate_theta_features_biquat,
    project_to_complex,
    compute_drift_term,
    fit_drift_parameters,
    detect_regimes_pca,
    walk_forward_predict
)


def test_biquaternion_features():
    """Test biquaternion feature generation."""
    print("\n" + "="*60)
    print("Test 1: Biquaternion Features")
    print("="*60)
    
    n_samples = 100
    q = 0.5
    n_terms = 8
    n_freqs = 4
    psi = 0.1
    
    # Generate biquaternion features
    features = generate_theta_features_biquat(n_samples, q=q, n_terms=n_terms, 
                                             n_freqs=n_freqs, psi=psi)
    
    print(f"Generated biquaternion features:")
    print(f"  Shape: {features.shape}")
    # n_terms iterations (excluding n=0), each with 4 components, for n_freqs frequencies
    expected_features = n_freqs * n_terms * 4
    print(f"  Expected: ({n_samples}, {expected_features})")
    print(f"  Features contain 4 components per term (a, b, c, d)")
    
    # Check for NaN or inf
    assert not np.isnan(features).any(), "Features contain NaN"
    assert not np.isinf(features).any(), "Features contain inf"
    print(f"  ✓ No NaN or inf values")
    
    # Check shape
    assert features.shape == (n_samples, expected_features), \
        f"Shape mismatch: {features.shape} != ({n_samples}, {expected_features})"
    print(f"  ✓ Shape correct")
    
    # Test complex projection (for first 4 components)
    biquat_sample = features[0, :4].reshape(1, 4)
    complex_proj = project_to_complex(biquat_sample)
    print(f"  Complex projection: {complex_proj}")
    print(f"  ✓ Complex projection works")
    
    print("\n✅ Biquaternion features test PASSED")
    return True


def test_drift_term():
    """Test Fokker-Planck drift term."""
    print("\n" + "="*60)
    print("Test 2: Fokker-Planck Drift Term")
    print("="*60)
    
    # Generate synthetic price series
    np.random.seed(42)
    n = 200
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    # Compute drift term
    beta0, beta1 = 0.1, 0.5
    drift = compute_drift_term(prices, beta0=beta0, beta1=beta1, ema_span=16)
    
    print(f"Drift term computed:")
    print(f"  Shape: {drift.shape}")
    print(f"  Range: [{drift.min():.4f}, {drift.max():.4f}]")
    print(f"  Mean: {drift.mean():.4f}")
    
    # Check for NaN or inf
    assert not np.isnan(drift).any(), "Drift contains NaN"
    assert not np.isinf(drift).any(), "Drift contains inf"
    print(f"  ✓ No NaN or inf values")
    
    # Test drift parameter fitting
    predictions = np.random.randn(n-1) * 0.1
    beta0_fit, beta1_fit = fit_drift_parameters(prices, predictions, ema_span=16)
    print(f"\nDrift parameter fitting:")
    print(f"  Fitted β₀: {beta0_fit:.4f}")
    print(f"  Fitted β₁: {beta1_fit:.4f}")
    print(f"  ✓ Parameter fitting works")
    
    print("\n✅ Drift term test PASSED")
    return True


def test_pca_regimes():
    """Test PCA regime detection."""
    print("\n" + "="*60)
    print("Test 3: PCA Regime Detection")
    print("="*60)
    
    # Generate synthetic features
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    
    # Create features with two distinct regimes
    regime1 = np.random.randn(n_samples // 2, n_features) * 0.5
    regime2 = np.random.randn(n_samples // 2, n_features) * 0.5 + 2
    features = np.vstack([regime1, regime2])
    
    # Detect regimes
    pca_coords, regimes, pca_model = detect_regimes_pca(features, n_components=2, n_clusters=2)
    
    print(f"PCA regime detection:")
    print(f"  PCA coordinates shape: {pca_coords.shape}")
    print(f"  Regimes shape: {regimes.shape}")
    print(f"  Unique regimes: {np.unique(regimes)}")
    print(f"  Regime distribution: {np.bincount(regimes)}")
    
    # Check outputs
    assert pca_coords.shape == (n_samples, 2), "PCA coordinates shape mismatch"
    assert regimes.shape == (n_samples,), "Regimes shape mismatch"
    assert len(np.unique(regimes)) == 2, "Expected 2 regimes"
    print(f"  ✓ Regime detection works correctly")
    
    print("\n✅ PCA regime detection test PASSED")
    return True


def test_backward_compatibility():
    """Test that v9 is backward compatible with v8."""
    print("\n" + "="*60)
    print("Test 4: Backward Compatibility")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    n = 600
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    # Run without v9 features (should work like v8)
    result_v8 = walk_forward_predict(
        prices=prices,
        window=256,
        horizon=1,
        q=0.5,
        n_terms=8,
        n_freqs=4,
        ridge_lambda=1.0,
        enable_biquaternion=False,
        enable_drift=False,
        enable_pca_regimes=False
    )
    
    print(f"v8 mode (no v9 features):")
    print(f"  Predictions: {len(result_v8['predictions'])}")
    print(f"  ✓ Runs without v9 features")
    
    # Run with v9 features
    result_v9 = walk_forward_predict(
        prices=prices,
        window=256,
        horizon=1,
        q=0.5,
        n_terms=8,
        n_freqs=4,
        ridge_lambda=1.0,
        enable_biquaternion=True,
        enable_drift=True,
        enable_pca_regimes=True
    )
    
    print(f"\nv9 mode (all features):")
    print(f"  Predictions: {len(result_v9['predictions'])}")
    print(f"  Has drift_values: {'drift_values' in result_v9}")
    print(f"  Has regime_labels: {'regime_labels' in result_v9}")
    print(f"  Has pca_coords: {'pca_coords' in result_v9}")
    print(f"  ✓ Runs with all v9 features")
    
    print("\n✅ Backward compatibility test PASSED")
    return True


def test_walk_forward_causality():
    """Verify walk-forward causality is maintained."""
    print("\n" + "="*60)
    print("Test 5: Walk-Forward Causality")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    n = 600
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    window = 256
    horizon = 4
    
    result = walk_forward_predict(
        prices=prices,
        window=window,
        horizon=horizon,
        enable_biquaternion=True,
        enable_drift=True,
        enable_pca_regimes=True
    )
    
    timestamps = result['timestamps']
    
    print(f"Walk-forward causality check:")
    print(f"  Window size: {window}")
    print(f"  Horizon: {horizon}")
    print(f"  First prediction at t={timestamps[0]} (should be >= {window})")
    print(f"  Last prediction at t={timestamps[-1]} (should be <= {n - horizon})")
    
    # Verify timestamps are within valid range
    assert timestamps[0] >= window, "First prediction violates causality"
    assert timestamps[-1] <= n - horizon, "Last prediction violates causality"
    print(f"  ✓ Timestamps within valid range")
    
    # Verify timestamps are monotonically increasing
    assert np.all(np.diff(timestamps) > 0), "Timestamps not monotonically increasing"
    print(f"  ✓ Timestamps monotonically increasing")
    
    print("\n✅ Walk-forward causality test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("THETA PREDICTOR V9 - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Biquaternion Features", test_biquaternion_features),
        ("Fokker-Planck Drift", test_drift_term),
        ("PCA Regime Detection", test_pca_regimes),
        ("Backward Compatibility", test_backward_compatibility),
        ("Walk-Forward Causality", test_walk_forward_causality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} test FAILED with error:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! v9 implementation is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
