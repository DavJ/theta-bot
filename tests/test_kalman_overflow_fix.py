#!/usr/bin/env python
"""
Unit test for Kalman overflow fix with large r_hat values.

Verifies that the r_max clipping prevents exp(r_hat) overflow.
"""

import numpy as np
import pandas as pd
import warnings

from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy, DualKalmanParams


BASE_PRICE = 50000  # Base price for test data generation


def test_kalman_no_overflow_with_large_values():
    """Test that large r_hat values don't cause overflow."""
    print("Testing Kalman strategy with extreme values...")
    
    # Create strategy with default r_max=8.0
    strategy = MeanRevDualKalmanStrategy()
    
    # Create test data that might cause large r_hat values
    n = 100
    np.random.seed(42)
    
    # Simulate price data with high volatility
    prices = BASE_PRICE * (1 + np.random.randn(n) * 0.1).cumprod()
    
    # Create features with extreme values
    features_df = pd.DataFrame({
        'close': prices,
        'C': np.random.uniform(0, 1, n),  # Concentration
        'psi': np.random.uniform(0, 2*np.pi, n),  # Phase
        'rv': np.random.uniform(0.01, 0.5, n),  # Realized volatility
        'risk_budget': np.ones(n),
    })
    
    # Catch warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Generate intent
        intent = strategy.generate_intent(features_df)
        
        # Check for overflow warnings
        overflow_warnings = [warning for warning in w if 'overflow' in str(warning.message).lower()]
        
        if overflow_warnings:
            print("❌ FAILED: Overflow warnings detected!")
            for warning in overflow_warnings:
                print(f"  Warning: {warning.message}")
            return False
        
    # Verify the intent is reasonable
    if not isinstance(intent.desired_exposure, (int, float)):
        print("❌ FAILED: desired_exposure is not a number!")
        return False
    
    if not np.isfinite(intent.desired_exposure):
        print("❌ FAILED: desired_exposure is not finite!")
        return False
    
    print(f"✓ No overflow warnings detected")
    print(f"✓ desired_exposure = {intent.desired_exposure:.4f}")
    
    # Test generate_series as well
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        series = strategy.generate_series(features_df)
        
        overflow_warnings = [warning for warning in w if 'overflow' in str(warning.message).lower()]
        
        if overflow_warnings:
            print("❌ FAILED: Overflow warnings in generate_series!")
            for warning in overflow_warnings:
                print(f"  Warning: {warning.message}")
            return False
    
    if not all(np.isfinite(series)):
        print("❌ FAILED: generate_series produced non-finite values!")
        return False
    
    print(f"✓ generate_series: all values finite")
    print(f"✓ generate_series: range [{series.min():.4f}, {series.max():.4f}]")
    
    return True


def test_r_max_parameter():
    """Test that r_max parameter is properly used."""
    print("\nTesting r_max parameter...")
    
    # Test with default r_max
    params_default = DualKalmanParams()
    assert params_default.r_max == 8.0, "Default r_max should be 8.0"
    print(f"✓ Default r_max = {params_default.r_max}")
    
    # Test with custom r_max
    params_custom = DualKalmanParams(r_max=5.0)
    assert params_custom.r_max == 5.0, "Custom r_max should be 5.0"
    print(f"✓ Custom r_max = {params_custom.r_max}")
    
    # Verify exp(r_max) is reasonable
    max_scale = np.exp(params_default.r_max)
    print(f"✓ exp(r_max) = {max_scale:.2f} (should be ~2981)")
    
    # Verify this is within s_max bounds
    assert max_scale > params_default.s_max, "exp(r_max) should exceed s_max for clipping to work"
    print(f"✓ exp(r_max) > s_max ({max_scale:.2f} > {params_default.s_max})")
    
    return True


def main():
    print("=" * 60)
    print("Kalman Overflow Fix Unit Tests")
    print("=" * 60)
    
    all_passed = True
    
    try:
        if not test_r_max_parameter():
            all_passed = False
    except Exception as e:
        print(f"❌ test_r_max_parameter failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        if not test_kalman_no_overflow_with_large_values():
            all_passed = False
    except Exception as e:
        print(f"❌ test_kalman_no_overflow_with_large_values failed with exception: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    else:
        print("Some tests failed! ❌")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())
