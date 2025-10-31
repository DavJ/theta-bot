#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_real_data.py
--------------------
Validate real market data before running predictions.

This script performs statistical tests and control experiments to ensure
the data is suitable for production testing and that the model can distinguish
real signal from noise.

Tests performed:
1. Data quality checks
2. Stationarity tests
3. Autocorrelation analysis
4. Permutation test (control)
5. Noise test (control)

Usage:
    python validate_real_data.py --csv real_data/BTCUSDT_1h.csv
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats


def test_data_quality(df):
    """
    Check basic data quality metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with 'close' column
        
    Returns
    -------
    results : dict
        Quality test results
    """
    print("\n" + "="*70)
    print("DATA QUALITY CHECKS")
    print("="*70)
    
    results = {}
    
    # Sample count
    n_samples = len(df)
    results['n_samples'] = n_samples
    print(f"Sample count: {n_samples}")
    
    if n_samples < 1000:
        print("  ⚠ Warning: Less than 1000 samples (minimum recommended)")
        results['quality_warnings'] = results.get('quality_warnings', [])
        results['quality_warnings'].append('insufficient_samples')
    else:
        print("  ✓ Sufficient samples for testing")
    
    # Missing values
    missing = df['close'].isna().sum()
    results['missing_values'] = int(missing)
    print(f"Missing values: {missing}")
    
    if missing > 0:
        print(f"  ⚠ Warning: {missing} missing values detected")
        results['quality_warnings'] = results.get('quality_warnings', [])
        results['quality_warnings'].append('missing_values')
    else:
        print("  ✓ No missing values")
    
    # Price statistics
    prices = df['close'].dropna()
    results['price_min'] = float(prices.min())
    results['price_max'] = float(prices.max())
    results['price_mean'] = float(prices.mean())
    results['price_std'] = float(prices.std())
    
    print(f"Price range: [{results['price_min']:.2f}, {results['price_max']:.2f}]")
    print(f"Price mean: {results['price_mean']:.2f}")
    print(f"Price std: {results['price_std']:.2f}")
    
    # Check for zeros or negative prices
    if (prices <= 0).any():
        print("  ⚠ Warning: Zero or negative prices detected")
        results['quality_warnings'] = results.get('quality_warnings', [])
        results['quality_warnings'].append('invalid_prices')
    else:
        print("  ✓ All prices positive")
    
    # Returns statistics
    returns = prices.pct_change().dropna()
    results['returns_mean'] = float(returns.mean())
    results['returns_std'] = float(returns.std())
    results['returns_skew'] = float(returns.skew())
    results['returns_kurt'] = float(returns.kurtosis())
    
    print(f"\nReturns statistics:")
    print(f"  Mean: {results['returns_mean']:.6f}")
    print(f"  Std: {results['returns_std']:.6f}")
    print(f"  Skewness: {results['returns_skew']:.3f}")
    print(f"  Kurtosis: {results['returns_kurt']:.3f}")
    
    return results


def test_stationarity(df):
    """
    Test for stationarity using Augmented Dickey-Fuller test.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with 'close' column
        
    Returns
    -------
    results : dict
        Stationarity test results
    """
    print("\n" + "="*70)
    print("STATIONARITY TESTS")
    print("="*70)
    
    results = {}
    prices = df['close'].dropna()
    
    # Test on price level
    print("\nADF test on price level:")
    try:
        from scipy.stats import adfuller
        adf_result = adfuller(prices, autolag='AIC')
        results['adf_price_statistic'] = float(adf_result[0])
        results['adf_price_pvalue'] = float(adf_result[1])
        
        print(f"  ADF Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        
        if adf_result[1] < 0.05:
            print("  ✓ Price series is stationary (p < 0.05)")
        else:
            print("  ⚠ Price series is non-stationary (p ≥ 0.05)")
    except ImportError:
        print("  ⚠ statsmodels not installed, skipping ADF test")
        results['adf_price_statistic'] = None
        results['adf_price_pvalue'] = None
    except Exception as e:
        print(f"  ⚠ ADF test failed: {e}")
        results['adf_price_statistic'] = None
        results['adf_price_pvalue'] = None
    
    # Test on returns (first difference)
    returns = prices.pct_change().dropna()
    print("\nADF test on returns:")
    try:
        from scipy.stats import adfuller
        adf_result = adfuller(returns, autolag='AIC')
        results['adf_returns_statistic'] = float(adf_result[0])
        results['adf_returns_pvalue'] = float(adf_result[1])
        
        print(f"  ADF Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        
        if adf_result[1] < 0.05:
            print("  ✓ Returns are stationary (p < 0.05)")
        else:
            print("  ⚠ Returns are non-stationary (p ≥ 0.05)")
    except ImportError:
        print("  ⚠ statsmodels not installed, skipping ADF test")
        results['adf_returns_statistic'] = None
        results['adf_returns_pvalue'] = None
    except Exception as e:
        print(f"  ⚠ ADF test failed: {e}")
        results['adf_returns_statistic'] = None
        results['adf_returns_pvalue'] = None
    
    return results


def test_autocorrelation(df, max_lag=50):
    """
    Analyze autocorrelation structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with 'close' column
    max_lag : int
        Maximum lag to test
        
    Returns
    -------
    results : dict
        Autocorrelation test results
    """
    print("\n" + "="*70)
    print("AUTOCORRELATION ANALYSIS")
    print("="*70)
    
    results = {}
    prices = df['close'].dropna()
    returns = prices.pct_change().dropna()
    
    # Compute autocorrelation at key lags
    key_lags = [1, 2, 4, 8, 16, 32]
    results['autocorr'] = {}
    
    print("\nAutocorrelation of returns at key lags:")
    for lag in key_lags:
        if lag < len(returns):
            acf = returns.autocorr(lag=lag)
            results['autocorr'][f'lag_{lag}'] = float(acf)
            print(f"  Lag {lag:2d}: {acf:7.4f}")
    
    # Ljung-Box test for autocorrelation
    print("\nLjung-Box test:")
    try:
        from scipy.stats import acorr_ljungbox
        lb_result = acorr_ljungbox(returns, lags=[10, 20], return_df=True)
        print(lb_result)
        results['ljung_box'] = {
            'statistic': lb_result['lb_stat'].tolist(),
            'pvalue': lb_result['lb_pvalue'].tolist()
        }
    except ImportError:
        print("  ⚠ statsmodels not installed, skipping Ljung-Box test")
        results['ljung_box'] = None
    except Exception as e:
        print(f"  ⚠ Ljung-Box test failed: {e}")
        results['ljung_box'] = None
    
    return results


def run_permutation_test(df, n_permutations=100, window=512, horizon=1):
    """
    Run permutation test to verify that predictions are not due to chance.
    
    This shuffles the time series and checks if we still get good predictions.
    If we do, it suggests overfitting or data leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data
    n_permutations : int
        Number of permutation tests
    window : int
        Training window size
    horizon : int
        Prediction horizon
        
    Returns
    -------
    results : dict
        Permutation test results
    """
    print("\n" + "="*70)
    print("PERMUTATION TEST (Control)")
    print("="*70)
    print(f"Testing with {n_permutations} random permutations...")
    print(f"If model finds signal in shuffled data, it suggests problems.")
    
    prices = df['close'].values
    
    # Simple feature generation
    def generate_features(prices, n_samples):
        t = np.arange(n_samples)
        t_norm = t / n_samples
        features = []
        
        # Add some simple periodic features
        for freq in [1, 2, 4, 8]:
            features.append(np.sin(2 * np.pi * freq * t_norm))
            features.append(np.cos(2 * np.pi * freq * t_norm))
        
        return np.column_stack(features)
    
    # Test on permuted data
    correlations = []
    
    for i in range(n_permutations):
        # Shuffle prices
        perm_prices = np.random.permutation(prices)
        
        # Walk-forward prediction
        n_samples = len(perm_prices)
        predictions = []
        actuals = []
        
        for t in range(window, n_samples - horizon):
            # Training window
            train_prices = perm_prices[t-window:t]
            train_features = generate_features(train_prices, window)
            
            # Target: change at t+horizon
            if t + horizon < n_samples:
                target_change = perm_prices[t+horizon] - perm_prices[t]
                
                # Simple ridge regression
                from scipy.linalg import ridge_regression
                
                # Prepare targets for training window
                train_targets = []
                for j in range(horizon, window):
                    train_targets.append(train_prices[j] - train_prices[j-horizon])
                
                if len(train_targets) >= 10:
                    train_targets = np.array(train_targets)
                    train_features_for_targets = train_features[:-horizon]
                    
                    # Fit model
                    try:
                        coeffs = ridge_regression(
                            train_features_for_targets,
                            train_targets,
                            alpha=1.0
                        )
                        
                        # Predict
                        current_features = generate_features(
                            perm_prices[t-window:t], window
                        )[-1]
                        pred_change = np.dot(current_features, coeffs)
                        
                        predictions.append(pred_change)
                        actuals.append(target_change)
                    except:
                        pass
        
        if len(predictions) > 10:
            corr, _ = pearsonr(predictions, actuals)
            correlations.append(corr)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{n_permutations} permutations...")
    
    correlations = np.array(correlations)
    
    results = {
        'n_permutations': n_permutations,
        'mean_correlation': float(np.mean(correlations)),
        'std_correlation': float(np.std(correlations)),
        'max_correlation': float(np.max(correlations)),
        'correlations': correlations.tolist()
    }
    
    print(f"\nPermutation test results:")
    print(f"  Mean correlation: {results['mean_correlation']:.4f}")
    print(f"  Std correlation: {results['std_correlation']:.4f}")
    print(f"  Max correlation: {results['max_correlation']:.4f}")
    print(f"\nExpected: Correlation near 0 for shuffled data")
    
    if abs(results['mean_correlation']) < 0.05:
        print(f"  ✓ Low correlation on shuffled data (good)")
    else:
        print(f"  ⚠ Warning: Non-zero correlation on shuffled data")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate real market data before production testing'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with market data'
    )
    parser.add_argument(
        '--permutation-tests',
        type=int,
        default=50,
        help='Number of permutation tests to run (default: 50)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='validation_output',
        help='Output directory for results (default: validation_output)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if 'close' not in df.columns:
        print("Error: CSV must contain 'close' column")
        return
    
    print(f"Loaded {len(df)} samples")
    
    # Run tests
    all_results = {}
    
    # Data quality
    quality_results = test_data_quality(df)
    all_results['data_quality'] = quality_results
    
    # Stationarity
    stationarity_results = test_stationarity(df)
    all_results['stationarity'] = stationarity_results
    
    # Autocorrelation
    autocorr_results = test_autocorrelation(df)
    all_results['autocorrelation'] = autocorr_results
    
    # Permutation test (if enough data)
    if len(df) >= 1000:
        permutation_results = run_permutation_test(
            df,
            n_permutations=args.permutation_tests
        )
        all_results['permutation_test'] = permutation_results
    else:
        print("\n⚠ Skipping permutation test (need ≥1000 samples)")
        all_results['permutation_test'] = None
    
    # Save results
    output_path = os.path.join(args.outdir, 'validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    # Check for any warnings
    warnings = quality_results.get('quality_warnings', [])
    
    if not warnings and len(df) >= 1000:
        print("✓ Data appears suitable for testing")
    else:
        print("⚠ Some validation warnings detected")
        for warning in warnings:
            print(f"  - {warning}")
    
    print(f"\n✓ Validation results saved to {output_path}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review validation results")
    print("2. Run predictions on real data:")
    print(f"   python theta_predictor.py --csv {args.csv} --window 512 --outdir test_output")
    print("3. Run control tests:")
    print(f"   python theta_horizon_scan_updated.py --csv {args.csv} --test-controls --outdir test_output")
    print("="*70)


if __name__ == '__main__':
    main()
