#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_hyperparameters.py
---------------------------
Optimize theta model hyperparameters for real market data.

This script performs grid search over key hyperparameters:
- q: Modular parameter
- n_terms: Number of theta series terms
- n_freqs: Number of frequency components
- lambda: Ridge regression regularization

Uses walk-forward validation to prevent overfitting.

Usage:
    python optimize_hyperparameters.py --csv real_data/BTCUSDT_1h.csv
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import product
import time


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
    
    # Theta-inspired features
    for m in range(-n_terms, n_terms + 1):
        # Imaginary time component
        psi = 2 * np.pi * q * m * t_norm
        
        for freq in range(1, n_freqs + 1):
            omega = 2 * np.pi * freq * t_norm
            
            # Complex exponential decomposed into sin/cos
            features.append(np.sin(omega + psi))
            features.append(np.cos(omega + psi))
    
    return np.column_stack(features)


def walk_forward_predict(prices, window, horizon, q, n_terms, n_freqs, ridge_lambda):
    """
    Perform walk-forward prediction with given hyperparameters.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
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
        Ridge regularization parameter
        
    Returns
    -------
    correlation : float
        Pearson correlation between predictions and actuals
    hit_rate : float
        Directional accuracy (0 to 1)
    """
    n_samples = len(prices)
    predictions = []
    actuals = []
    
    for t in range(window, n_samples - horizon):
        # Training window
        train_prices = prices[t-window:t]
        train_features = generate_theta_features(window, q, n_terms, n_freqs)
        
        # Prepare training targets
        train_targets = []
        for j in range(horizon, window):
            train_targets.append(train_prices[j] - train_prices[j-horizon])
        
        if len(train_targets) < 10:
            continue
            
        train_targets = np.array(train_targets)
        train_features_for_targets = train_features[:-horizon]
        
        # Ridge regression
        try:
            # Add ridge regularization
            X = train_features_for_targets
            y = train_targets
            
            # Closed-form ridge solution: (X^T X + λI)^{-1} X^T y
            XtX = X.T @ X
            ridge_term = ridge_lambda * np.eye(XtX.shape[0])
            coeffs = np.linalg.solve(XtX + ridge_term, X.T @ y)
            
            # Current features for prediction
            current_features = generate_theta_features(window, q, n_terms, n_freqs)[-1]
            pred_change = np.dot(current_features, coeffs)
            
            # Actual change
            actual_change = prices[t + horizon] - prices[t]
            
            predictions.append(pred_change)
            actuals.append(actual_change)
            
        except np.linalg.LinAlgError:
            continue
    
    if len(predictions) < 10:
        return 0.0, 0.5
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Compute metrics
    try:
        correlation, _ = pearsonr(predictions, actuals)
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    # Directional hit rate
    correct_direction = np.sum(np.sign(predictions) == np.sign(actuals))
    hit_rate = correct_direction / len(predictions)
    
    return correlation, hit_rate


def optimize_hyperparameters(df, window=512, horizon=1, val_split=0.3):
    """
    Grid search over hyperparameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with 'close' column
    window : int
        Training window size
    horizon : int
        Prediction horizon
    val_split : float
        Fraction of data to use for validation
        
    Returns
    -------
    results : pd.DataFrame
        Results for all parameter combinations
    best_params : dict
        Best hyperparameters
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    prices = df['close'].values
    
    # Split into train and validation
    n_samples = len(prices)
    val_size = int(n_samples * val_split)
    train_size = n_samples - val_size
    
    train_prices = prices[:train_size]
    val_prices = prices[train_size:]
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Window: {window}")
    print(f"Horizon: {horizon}")
    
    # Define parameter grid
    param_grid = {
        'q': [0.3, 0.5, 0.7],
        'n_terms': [8, 12, 16],
        'n_freqs': [4, 6, 8],
        'ridge_lambda': [0.1, 1.0, 10.0]
    }
    
    print(f"\nParameter grid:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    print(f"\nTotal combinations: {len(combinations)}")
    print("Testing on validation set...\n")
    
    # Test each combination
    results = []
    
    for i, params in enumerate(combinations):
        param_dict = dict(zip(param_names, params))
        
        # Test on validation set
        try:
            val_corr, val_hit = walk_forward_predict(
                val_prices,
                window,
                horizon,
                param_dict['q'],
                param_dict['n_terms'],
                param_dict['n_freqs'],
                param_dict['ridge_lambda']
            )
            
            results.append({
                **param_dict,
                'val_correlation': val_corr,
                'val_hit_rate': val_hit,
                'val_score': val_corr  # Using correlation as score
            })
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"  Tested {i+1}/{len(combinations)} combinations...")
                print(f"    Current: q={param_dict['q']:.1f}, "
                      f"n_terms={param_dict['n_terms']}, "
                      f"n_freqs={param_dict['n_freqs']}, "
                      f"λ={param_dict['ridge_lambda']:.1f}")
                print(f"    Val corr={val_corr:.4f}, hit={val_hit:.1%}")
        
        except Exception as e:
            print(f"  Error with params {param_dict}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    # Find best parameters
    best_idx = results_df['val_score'].idxmax()
    best_params = results_df.loc[best_idx].to_dict()
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\nBest parameters:")
    print(f"  q = {best_params['q']:.2f}")
    print(f"  n_terms = {int(best_params['n_terms'])}")
    print(f"  n_freqs = {int(best_params['n_freqs'])}")
    print(f"  lambda = {best_params['ridge_lambda']:.2f}")
    
    print(f"\nBest validation performance:")
    print(f"  Correlation: {best_params['val_correlation']:.4f}")
    print(f"  Hit rate: {best_params['val_hit_rate']:.1%}")
    
    # Show top 5 configurations
    print(f"\nTop 5 configurations:")
    top5 = results_df.nlargest(5, 'val_score')
    print(top5[['q', 'n_terms', 'n_freqs', 'ridge_lambda', 
                'val_correlation', 'val_hit_rate']].to_string(index=False))
    
    return results_df, best_params


def test_best_params_on_full_data(df, best_params, window=512, horizons=[1, 2, 4, 8]):
    """
    Test best parameters on full dataset with multiple horizons.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data
    best_params : dict
        Best hyperparameters
    window : int
        Training window size
    horizons : list
        Prediction horizons to test
        
    Returns
    -------
    results : pd.DataFrame
        Results for each horizon
    """
    print("\n" + "="*70)
    print("TESTING BEST PARAMETERS ON FULL DATA")
    print("="*70)
    
    prices = df['close'].values
    results = []
    
    for horizon in horizons:
        print(f"\nTesting horizon h={horizon}...")
        
        corr, hit_rate = walk_forward_predict(
            prices,
            window,
            horizon,
            best_params['q'],
            int(best_params['n_terms']),
            int(best_params['n_freqs']),
            best_params['ridge_lambda']
        )
        
        results.append({
            'horizon': horizon,
            'correlation': corr,
            'hit_rate': hit_rate
        })
        
        print(f"  Correlation: {corr:.4f}")
        print(f"  Hit rate: {hit_rate:.1%}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Optimize theta model hyperparameters for real data'
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to CSV file with market data'
    )
    parser.add_argument(
        '--window',
        type=int,
        default=512,
        help='Training window size (default: 512)'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=1,
        help='Primary prediction horizon for optimization (default: 1)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.3,
        help='Validation set size (default: 0.3)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='optimization_output',
        help='Output directory (default: optimization_output)'
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
    
    if len(df) < args.window + 100:
        print(f"Error: Need at least {args.window + 100} samples")
        return
    
    # Run optimization
    start_time = time.time()
    results_df, best_params = optimize_hyperparameters(
        df,
        window=args.window,
        horizon=args.horizon,
        val_split=args.val_split
    )
    elapsed = time.time() - start_time
    
    print(f"\nOptimization completed in {elapsed:.1f} seconds")
    
    # Save optimization results
    results_path = os.path.join(args.outdir, 'optimization_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Saved optimization results to {results_path}")
    
    # Save best parameters
    best_params_clean = {
        'q': float(best_params['q']),
        'n_terms': int(best_params['n_terms']),
        'n_freqs': int(best_params['n_freqs']),
        'ridge_lambda': float(best_params['ridge_lambda']),
        'validation_correlation': float(best_params['val_correlation']),
        'validation_hit_rate': float(best_params['val_hit_rate'])
    }
    
    best_params_path = os.path.join(args.outdir, 'best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params_clean, f, indent=2)
    print(f"✓ Saved best parameters to {best_params_path}")
    
    # Test on full data with multiple horizons
    print("\n" + "="*70)
    print("Testing best parameters on multiple horizons...")
    horizon_results = test_best_params_on_full_data(
        df,
        best_params,
        window=args.window,
        horizons=[1, 2, 4, 8, 16, 32]
    )
    
    horizon_results_path = os.path.join(args.outdir, 'horizon_test_results.csv')
    horizon_results.to_csv(horizon_results_path, index=False)
    print(f"\n✓ Saved horizon test results to {horizon_results_path}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(horizon_results['horizon'], horizon_results['correlation'], 'o-')
    ax1.set_xlabel('Prediction Horizon')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Correlation vs Horizon (Optimized Params)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    ax2.plot(horizon_results['horizon'], horizon_results['hit_rate'], 'o-')
    ax2.set_xlabel('Prediction Horizon')
    ax2.set_ylabel('Hit Rate')
    ax2.set_title('Hit Rate vs Horizon (Optimized Params)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, 'optimized_performance.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_path}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review optimization results")
    print("2. Use optimized parameters in production:")
    print(f"   python theta_predictor.py --csv {args.csv} \\")
    print(f"       --window {args.window} \\")
    print(f"       --q {best_params_clean['q']:.2f} \\")
    print(f"       --n-terms {best_params_clean['n_terms']} \\")
    print(f"       --n-freqs {best_params_clean['n_freqs']} \\")
    print(f"       --lambda {best_params_clean['ridge_lambda']:.2f}")
    print("="*70)


if __name__ == '__main__':
    main()
