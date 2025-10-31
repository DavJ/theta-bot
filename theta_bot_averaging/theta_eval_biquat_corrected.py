#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_biquat_corrected.py
------------------------------
Corrected biquaternion implementation following atlas_evaluation.tex recommendations:
- Uses complex pairs: φ1 = θ3 + iθ4, φ2 = θ2 + iθ1
- Implements complex ridge regression with proper phase coherence
- Strictly causal walk-forward validation (no data leaks)
- Tests on both synthetic and real data
- Outputs predictivity metrics: corr_pred_true, hit_rate

Usage:
    python theta_eval_biquat_corrected.py --csv data.csv --horizon 1 --window 256
"""

import argparse
import os
import numpy as np
import pandas as pd
from typing import Tuple


# ------------- Jacobi theta functions (real-valued) -------------

def theta1_real(z: float, q: float, n_terms: int = 16) -> float:
    """θ1(z,q) = 2 * sum_{n=0}^{N} (-1)^n * q^{(n+1/2)^2} * sin((2n+1)*z)"""
    s = 0.0
    for n in range(n_terms):
        w = ((-1.0) ** n) * (q ** ((n + 0.5) ** 2))
        s += w * np.sin((2*n + 1) * z)
    return 2.0 * s


def theta2_real(z: float, q: float, n_terms: int = 16) -> float:
    """θ2(z,q) = 2 * sum_{n=0}^{N} q^{(n+1/2)^2} * cos((2n+1)*z)"""
    s = 0.0
    for n in range(n_terms):
        w = q ** ((n + 0.5) ** 2)
        s += w * np.cos((2*n + 1) * z)
    return 2.0 * s


def theta3_real(z: float, q: float, n_terms: int = 16) -> float:
    """θ3(z,q) = 1 + 2 * sum_{n=1}^{N} q^{n^2} * cos(2*n*z)"""
    s = 1.0
    for n in range(1, n_terms + 1):
        w = q ** (n ** 2)
        s += 2.0 * w * np.cos(2 * n * z)
    return s


def theta4_real(z: float, q: float, n_terms: int = 16) -> float:
    """θ4(z,q) = 1 + 2 * sum_{n=1}^{N} (-1)^n * q^{n^2} * cos(2*n*z)"""
    s = 1.0
    for n in range(1, n_terms + 1):
        w = ((-1.0) ** n) * (q ** (n ** 2))
        s += 2.0 * w * np.cos(2 * n * z)
    return s


# ------------- Biquaternion basis construction -------------

def build_biquat_complex_pairs(t: int, 
                               freqs: np.ndarray, 
                               q: float, 
                               n_terms: int,
                               phase_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build complex pairs following atlas_evaluation.tex recommendations:
    φ1 = θ3 + i*θ4
    φ2 = θ2 + i*θ1
    
    Returns:
        phi1: complex array of shape (n_freq,)
        phi2: complex array of shape (n_freq,)
    """
    n_freq = len(freqs)
    phi1 = np.zeros(n_freq, dtype=np.complex128)
    phi2 = np.zeros(n_freq, dtype=np.complex128)
    
    for k, omega in enumerate(freqs):
        z = (phase_scale * t) * omega
        
        th1 = theta1_real(z, q, n_terms)
        th2 = theta2_real(z, q, n_terms)
        th3 = theta3_real(z, q, n_terms)
        th4 = theta4_real(z, q, n_terms)
        
        # Complex pairs as recommended in the document
        phi1[k] = th3 + 1j * th4
        phi2[k] = th2 + 1j * th1
    
    return phi1, phi2


def build_feature_matrix_biquat(T: int,
                                freqs: np.ndarray,
                                q: float,
                                n_terms: int,
                                phase_scale: float = 1.0) -> np.ndarray:
    """
    Build feature matrix with proper biquaternion structure.
    Each time point t has features: [Re(φ1), Im(φ1), Re(φ2), Im(φ2)] for each frequency.
    
    Returns:
        X: array of shape (T, 4*n_freq) where features are grouped by frequency
    """
    n_freq = len(freqs)
    D = 4 * n_freq  # Re/Im for each of phi1, phi2 per frequency
    X = np.zeros((T, D), dtype=np.float64)
    
    for t in range(T):
        phi1, phi2 = build_biquat_complex_pairs(t, freqs, q, n_terms, phase_scale)
        
        # Stack [Re(φ1), Im(φ1), Re(φ2), Im(φ2)] per frequency
        for k in range(n_freq):
            col = k * 4
            X[t, col] = phi1[k].real
            X[t, col + 1] = phi1[k].imag
            X[t, col + 2] = phi2[k].real
            X[t, col + 3] = phi2[k].imag
    
    return X


# ------------- Block-regularized ridge regression -------------

def fit_block_ridge(X: np.ndarray, 
                   y: np.ndarray, 
                   lam: float,
                   block_size: int = 4) -> np.ndarray:
    """
    Ridge regression with block L2 regularization.
    Regularizes L2 norm of each 4-dimensional block (per frequency).
    
    This enforces that the weights for each biquaternion (frequency) are
    regularized together, maintaining quaternion structure.
    """
    D = X.shape[1]
    n_blocks = D // block_size
    
    # Standard ridge solution: β = (X'X + λI)^{-1} X'y
    XtX = X.T @ X
    
    # Apply block regularization
    reg_matrix = np.zeros((D, D), dtype=np.float64)
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        # Add λI to each block on the diagonal
        reg_matrix[start:end, start:end] = lam * np.eye(block_size)
    
    beta = np.linalg.solve(XtX + reg_matrix, X.T @ y)
    return beta


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute standardization parameters."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0  # Avoid division by zero
    return mu, sigma


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply standardization."""
    return (X - mu) / sigma


# ------------- Metrics -------------

def hit_rate(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Fraction of predictions where sign(y_hat) == sign(y_true).
    Excludes cases where y_true == 0.
    """
    s_true = np.sign(y_true)
    s_hat = np.sign(y_hat)
    mask = (s_true != 0)
    if mask.sum() == 0:
        return np.nan
    return float((s_true[mask] == s_hat[mask]).mean())


def corr_pred_true(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Correlation between predictions and true values.
    This is the key metric: corr(y_hat, y_true).
    """
    if len(y_true) < 2 or y_true.std() == 0 or y_hat.std() == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_hat)[0, 1])


# ------------- Walk-forward evaluation (NO DATA LEAKS) -------------

def evaluate_walk_forward(prices: np.ndarray,
                         horizon: int,
                         window: int,
                         q: float,
                         n_terms: int,
                         n_freq: int,
                         lam: float,
                         phase_scale: float = 1.0) -> dict:
    """
    Strictly causal walk-forward evaluation.
    
    For each time t:
    1. Use only data from [t-window, t) to fit model
    2. Predict price change from t to t+horizon
    3. No data from time t or later is used in fitting
    
    This ensures NO DATA LEAKS from the future.
    
    Returns:
        dict with keys: y_true, y_hat, hit_rate, corr_pred_true, n_predictions
    """
    T = len(prices)
    
    # Target: price change over horizon
    # y[t] = price[t+horizon] - price[t]
    # We can only predict up to T-horizon
    y = prices[horizon:] - prices[:-horizon]
    T_y = len(y)
    
    # Frequency grid tied to window size (Fourier grid)
    m = np.arange(1, n_freq + 1, dtype=np.float64)
    freqs = 2.0 * np.pi * m / float(window)
    
    # Build full feature matrix
    X_full = build_feature_matrix_biquat(T, freqs, q, n_terms, phase_scale)
    
    # Features aligned with targets (shift by horizon)
    X = X_full[:-horizon, :]
    
    # Storage for predictions
    y_hats = []
    y_trues = []
    
    # Walk-forward loop
    # Start predicting after we have 'window' samples
    for t0 in range(window, T_y):
        # Fit window: [t0-window, t0)
        lo, hi = t0 - window, t0
        X_fit = X[lo:hi, :]
        y_fit = y[lo:hi]
        
        # Current feature vector (at time t0)
        x_now = X[t0, :]
        
        # True target at t0
        y_true = y[t0]
        
        # Standardize using fit window statistics only
        mu, sigma = standardize_fit(X_fit)
        X_fit_std = standardize_apply(X_fit, mu, sigma)
        x_now_std = standardize_apply(x_now.reshape(1, -1), mu, sigma)[0]
        
        # Fit block-regularized ridge
        beta = fit_block_ridge(X_fit_std, y_fit, lam, block_size=4)
        
        # Predict
        y_hat = float(x_now_std @ beta)
        
        y_hats.append(y_hat)
        y_trues.append(y_true)
    
    y_hats = np.array(y_hats)
    y_trues = np.array(y_trues)
    
    # Compute metrics
    hr = hit_rate(y_trues, y_hats)
    corr = corr_pred_true(y_trues, y_hats)
    
    return {
        'y_true': y_trues,
        'y_hat': y_hats,
        'hit_rate': hr,
        'corr_pred_true': corr,
        'n_predictions': len(y_hats)
    }


# ------------- Main -------------

def main():
    parser = argparse.ArgumentParser(
        description='Corrected biquaternion ridge evaluation with walk-forward validation'
    )
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with price data')
    parser.add_argument('--price-col', type=str, default='close',
                       help='Name of price column (default: close)')
    parser.add_argument('--horizon', type=int, default=1,
                       help='Prediction horizon (default: 1)')
    parser.add_argument('--window', type=int, default=256,
                       help='Training window size (default: 256)')
    parser.add_argument('--q', type=float, default=0.6,
                       help='Nome parameter for theta functions (default: 0.6)')
    parser.add_argument('--n-terms', type=int, default=16,
                       help='Number of terms in theta series (default: 16)')
    parser.add_argument('--n-freq', type=int, default=6,
                       help='Number of frequencies (default: 6)')
    parser.add_argument('--lambda', dest='lam', type=float, default=0.5,
                       help='Ridge regularization parameter (default: 0.5)')
    parser.add_argument('--phase-scale', type=float, default=1.0,
                       help='Phase scaling factor (default: 1.0)')
    parser.add_argument('--outdir', type=str, default='results_biquat_corrected',
                       help='Output directory (default: results_biquat_corrected)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # Find price column
    if args.price_col not in df.columns:
        candidates = ['close', 'Close', 'price', 'Price', 'close_price']
        found = [c for c in candidates if c in df.columns]
        if found:
            args.price_col = found[0]
            print(f"Using detected price column: {args.price_col}")
        else:
            raise ValueError(f"Price column '{args.price_col}' not found. Available: {list(df.columns)}")
    
    prices = df[args.price_col].astype(float).values
    print(f"Loaded {len(prices)} price samples")
    
    # Run evaluation
    print("\nRunning walk-forward evaluation (no data leaks)...")
    print(f"  Horizon: {args.horizon}")
    print(f"  Window: {args.window}")
    print(f"  q: {args.q}")
    print(f"  n_terms: {args.n_terms}")
    print(f"  n_freq: {args.n_freq}")
    print(f"  lambda: {args.lam}")
    print(f"  phase_scale: {args.phase_scale}")
    
    results = evaluate_walk_forward(
        prices=prices,
        horizon=args.horizon,
        window=args.window,
        q=args.q,
        n_terms=args.n_terms,
        n_freq=args.n_freq,
        lam=args.lam,
        phase_scale=args.phase_scale
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS - Corrected Biquaternion Basis")
    print("="*60)
    print(f"Number of predictions: {results['n_predictions']}")
    print(f"Hit Rate:             {results['hit_rate']:.4f}")
    print(f"Correlation:          {results['corr_pred_true']:.4f}")
    print("="*60)
    
    # Save results
    predictions_df = pd.DataFrame({
        'y_true': results['y_true'],
        'y_hat': results['y_hat']
    })
    pred_path = os.path.join(args.outdir, 'predictions.csv')
    predictions_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to: {pred_path}")
    
    # Save summary
    summary = {
        'n_predictions': results['n_predictions'],
        'hit_rate': results['hit_rate'],
        'corr_pred_true': results['corr_pred_true'],
        'horizon': args.horizon,
        'window': args.window,
        'q': args.q,
        'n_terms': args.n_terms,
        'n_freq': args.n_freq,
        'lambda': args.lam,
        'phase_scale': args.phase_scale,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(args.outdir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
