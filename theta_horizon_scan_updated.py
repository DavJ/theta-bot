#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_horizon_scan.py
---------------------
Evaluate how predictive correlation varies with prediction horizon.

This scans multiple horizons to identify resonance peaks where phase 
alignment matches modular symmetry, consistent with CCT/UBT predictions.

Outputs:
- theta_resonance.csv: Correlation vs. horizon data
- theta_resonance.png: Correlation vs. horizon plot

Author: Enhanced version based on COPILOT_BRIEF_v2.md
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def generate_theta_features(n_samples, q=0.5, n_terms=16, n_freqs=8):
    """
    Generate theta-based features for time series prediction.
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    q : float
        Modular parameter (0 < q < 1)
    n_terms : int
        Number of terms in theta series
    n_freqs : int
        Number of frequency components
        
    Returns
    -------
    features : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    """
    t = np.arange(n_samples)
    t_norm = t / n_samples
    
    features = []
    
    for k in range(n_freqs):
        omega = 0.5 + k * 0.3
        
        for n in range(-n_terms // 2, n_terms // 2 + 1):
            if n == 0:
                continue
            # Jacobi theta-like basis: e^(i*pi*n^2*q*t) * e^(2*pi*i*n*omega*t)
            phase = np.pi * n**2 * q * t_norm + 2 * np.pi * n * omega * t_norm
            features.append(np.cos(phase))
            features.append(np.sin(phase))
    
    return np.column_stack(features)


def theta_predict_horizon(prices, window, horizon, q=0.5, n_terms=16, 
                          n_freqs=8, ridge_lambda=1.0):
    """
    Predict at a specific horizon using theta basis with walk-forward validation.
    
    Parameters
    ----------
    prices : np.ndarray
        Price time series
    window : int
        Training window size
    horizon : int
        Prediction horizon
    q : float
        Modular parameter
    n_terms : int
        Number of theta terms
    n_freqs : int
        Number of frequencies
    ridge_lambda : float
        Ridge regularization
        
    Returns
    -------
    predictions : np.ndarray
        Predicted price deltas
    actuals : np.ndarray
        Actual price deltas
    """
    n = len(prices)
    
    if n < window + horizon + 10:
        return np.array([]), np.array([])
    
    deltas = np.diff(prices)
    
    predictions = []
    actuals = []
    
    for t in range(window, n - horizon):
        train_prices = prices[t - window : t]
        train_deltas = deltas[t - window - 1 : t - 1]
        
        # Generate features
        X_train = generate_theta_features(window, q=q, n_terms=n_terms, n_freqs=n_freqs)
        y_train = train_deltas
        
        # Standardize
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_train_std = (X_train - X_mean) / X_std
        
        # Ridge regression
        n_features = X_train_std.shape[1]
        try:
            XtX = X_train_std.T @ X_train_std
            reg_matrix = ridge_lambda * np.eye(n_features)
            beta = np.linalg.solve(XtX + reg_matrix, X_train_std.T @ y_train)
        except:
            continue
        
        # Predict
        X_pred_full = generate_theta_features(t + 1, q=q, n_terms=n_terms, n_freqs=n_freqs)
        X_pred = X_pred_full[t, :].reshape(1, -1)
        X_pred_std = (X_pred - X_mean) / X_std
        
        delta_pred = X_pred_std @ beta
        pred_value = delta_pred[0] * horizon
        
        actual_delta = prices[t + horizon] - prices[t]
        
        predictions.append(pred_value)
        actuals.append(actual_delta)
    
    return np.array(predictions), np.array(actuals)


def scan_horizons(prices, window, horizons, q=0.5, n_terms=16, n_freqs=8,
                 ridge_lambda=1.0, test_permutation=False, test_noise=False):
    """
    Scan multiple horizons and measure correlation.
    
    Parameters
    ----------
    prices : np.ndarray
        Price time series
    window : int
        Training window
    horizons : list of int
        Horizons to test
    q : float
        Modular parameter
    n_terms : int
        Number of theta terms
    n_freqs : int
        Number of frequencies
    ridge_lambda : float
        Ridge regularization
    test_permutation : bool
        If True, test with permuted prices (should give near-zero correlation)
    test_noise : bool
        If True, test with synthetic noise (should give near-zero correlation)
        
    Returns
    -------
    results : pd.DataFrame
        Results for each horizon
    """
    results = []
    
    # Test data
    if test_permutation:
        print("Testing with PERMUTED data (control test)...")
        test_prices = np.random.permutation(prices)
    elif test_noise:
        print("Testing with SYNTHETIC NOISE (control test)...")
        test_prices = np.random.randn(len(prices)) * np.std(prices) + np.mean(prices)
    else:
        test_prices = prices
    
    for h in horizons:
        print(f"  Testing horizon {h}...", end=' ')
        
        preds, actuals = theta_predict_horizon(
            test_prices, window, h, q=q, n_terms=n_terms, 
            n_freqs=n_freqs, ridge_lambda=ridge_lambda
        )
        
        if len(preds) > 2:
            corr, p_val = pearsonr(preds, actuals)
            
            # Directional accuracy
            hit_rate = np.mean(np.sign(preds) == np.sign(actuals))
            
            # RMS error
            rmse = np.sqrt(np.mean((preds - actuals) ** 2))
            
            results.append({
                'horizon': h,
                'correlation': corr,
                'correlation_pvalue': p_val,
                'hit_rate': hit_rate,
                'rmse': rmse,
                'n_samples': len(preds)
            })
            
            print(f"r={corr:.4f}, p={p_val:.4e}, hit={hit_rate:.3f}")
        else:
            print("insufficient data")
            results.append({
                'horizon': h,
                'correlation': np.nan,
                'correlation_pvalue': np.nan,
                'hit_rate': np.nan,
                'rmse': np.nan,
                'n_samples': 0
            })
    
    return pd.DataFrame(results)


def plot_resonance(results_df, outdir, title_suffix=""):
    """
    Plot correlation vs. horizon showing resonance peaks.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Correlation vs horizon
    ax = axes[0]
    horizons = results_df['horizon'].values
    corrs = results_df['correlation'].values
    
    ax.plot(horizons, corrs, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Correlation (r)')
    ax.set_title(f'Theta Projection: Correlation vs. Horizon{title_suffix}')
    ax.grid(True, alpha=0.3)
    
    # Log scale for x-axis
    if len(horizons) > 3:
        ax.set_xscale('log', base=2)
    
    # Hit rate vs horizon
    ax = axes[1]
    hit_rates = results_df['hit_rate'].values
    
    ax.plot(horizons, hit_rates, 's-', linewidth=2, markersize=8, color='green')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_xlabel('Prediction Horizon')
    ax.set_ylabel('Hit Rate')
    ax.set_title(f'Directional Accuracy vs. Horizon{title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if len(horizons) > 3:
        ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    suffix = title_suffix.lower().replace(' ', '_').replace('(', '').replace(')', '')
    filename = f'theta_resonance{suffix}.png'
    plt.savefig(os.path.join(outdir, filename), dpi=150)
    plt.close()
    print(f"Saved resonance plot to {outdir}/{filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Scan theta prediction correlation across horizons'
    )
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV with price data')
    parser.add_argument('--price-col', type=str, default='close',
                       help='Column name for price (default: close)')
    parser.add_argument('--window', type=int, default=512,
                       help='Training window size (default: 512)')
    parser.add_argument('--horizons', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16, 32, 64, 128],
                       help='Horizons to scan (default: 1 2 4 8 16 32 64 128)')
    parser.add_argument('--q', type=float, default=0.5,
                       help='Modular parameter (default: 0.5)')
    parser.add_argument('--n-terms', type=int, default=16,
                       help='Number of theta terms (default: 16)')
    parser.add_argument('--n-freqs', type=int, default=8,
                       help='Number of frequencies (default: 8)')
    parser.add_argument('--ridge-lambda', type=float, default=1.0,
                       help='Ridge regularization (default: 1.0)')
    parser.add_argument('--test-controls', action='store_true',
                       help='Run control tests (permutation and noise)')
    parser.add_argument('--outdir', type=str, default='theta_output',
                       help='Output directory (default: theta_output)')
    
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 60)
    print("Theta Horizon Resonance Scan")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  CSV: {args.csv}")
    print(f"  Price column: {args.price_col}")
    print(f"  Window: {args.window}")
    print(f"  Horizons: {args.horizons}")
    print(f"  q: {args.q}")
    print(f"  n_terms: {args.n_terms}")
    print(f"  n_freqs: {args.n_freqs}")
    print(f"  ridge_lambda: {args.ridge_lambda}")
    print()
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if args.price_col not in df.columns:
        raise ValueError(f"Column '{args.price_col}' not found")
    
    prices = df[args.price_col].values
    print(f"Loaded {len(prices)} samples")
    print()
    
    # Real data scan
    print("Scanning real data...")
    results_real = scan_horizons(
        prices=prices,
        window=args.window,
        horizons=args.horizons,
        q=args.q,
        n_terms=args.n_terms,
        n_freqs=args.n_freqs,
        ridge_lambda=args.ridge_lambda
    )
    print()
    
    # Save real results
    results_path = os.path.join(args.outdir, 'theta_resonance.csv')
    results_real.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    # Plot real results
    plot_resonance(results_real, args.outdir)
    
    # Control tests
    if args.test_controls:
        print("\nRunning control tests...")
        
        # Permutation test
        results_perm = scan_horizons(
            prices=prices,
            window=args.window,
            horizons=args.horizons,
            q=args.q,
            n_terms=args.n_terms,
            n_freqs=args.n_freqs,
            ridge_lambda=args.ridge_lambda,
            test_permutation=True
        )
        
        perm_path = os.path.join(args.outdir, 'theta_resonance_permutation.csv')
        results_perm.to_csv(perm_path, index=False)
        plot_resonance(results_perm, args.outdir, " (Permutation Control)")
        print()
        
        # Noise test
        results_noise = scan_horizons(
            prices=prices,
            window=args.window,
            horizons=args.horizons,
            q=args.q,
            n_terms=args.n_terms,
            n_freqs=args.n_freqs,
            ridge_lambda=args.ridge_lambda,
            test_noise=True
        )
        
        noise_path = os.path.join(args.outdir, 'theta_resonance_noise.csv')
        results_noise.to_csv(noise_path, index=False)
        plot_resonance(results_noise, args.outdir, " (Noise Control)")
    
    print("\n" + "=" * 60)
    print("Horizon scan complete!")
    print("=" * 60)
    
    # Summary
    print("\nSummary:")
    print("\nReal Data:")
    valid_corrs = results_real[~results_real['correlation'].isna()]
    if len(valid_corrs) > 0:
        max_corr_row = valid_corrs.loc[valid_corrs['correlation'].idxmax()]
        print(f"  Peak correlation: r={max_corr_row['correlation']:.4f} at horizon {int(max_corr_row['horizon'])}")
        print(f"  Average correlation: {valid_corrs['correlation'].mean():.4f}")
        print(f"  Average hit rate: {valid_corrs['hit_rate'].mean():.4f}")
    
    if args.test_controls:
        print("\nPermutation Control:")
        valid_perm = results_perm[~results_perm['correlation'].isna()]
        if len(valid_perm) > 0:
            print(f"  Average correlation: {valid_perm['correlation'].mean():.4f}")
            print(f"  Average hit rate: {valid_perm['hit_rate'].mean():.4f}")
        
        print("\nNoise Control:")
        valid_noise = results_noise[~results_noise['correlation'].isna()]
        if len(valid_noise) > 0:
            print(f"  Average correlation: {valid_noise['correlation'].mean():.4f}")
            print(f"  Average hit rate: {valid_noise['hit_rate'].mean():.4f}")


if __name__ == '__main__':
    main()
