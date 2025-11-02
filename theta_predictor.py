#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_predictor.py - v9
-----------------------
No-leak walk-forward prediction using theta basis projection.

Version 9 enhancements:
- Biquaternionic time support (τ = t + jψ)
- Fokker-Planck drift term
- PCA-based market regime detection
- Maintains strict walk-forward causality

This implements causal prediction where:
- Train window: W samples → predict horizon h
- Fit model only on [t-W, t) and predict at t+h
- Prevent lookahead bias
- Test multiple horizons: h = {1, 2, 4, 8, 16, 32}

Metrics measured:
- Pearson correlation between predicted and true deltas
- Directional hit rate (% correct sign)
- Cumulative trading PnL (long/short with transaction cost)

Author: Implementation based on COPILOT_BRIEF_v2.md and COPILOT_INSTRUCTIONS_V9.md
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ============================================================================
# Biquaternion Support (v9)
# ============================================================================

def project_to_complex(theta_biquat):
    """
    Project biquaternion coefficients to complex plane.
    
    θ_k = a_k + i*b_k + j*c_k + k*d_k
    Complex projection: a_k + i*b_k
    
    Parameters
    ----------
    theta_biquat : np.ndarray
        Biquaternion representation, shape (..., 4) where components are [a, b, c, d]
        
    Returns
    -------
    complex_proj : np.ndarray
        Complex projection, shape (...,) of dtype complex
    """
    return theta_biquat[..., 0] + 1j * theta_biquat[..., 1]


def generate_theta_features_biquat(n_samples, q=0.5, n_terms=16, n_freqs=8, psi=0.1):
    """
    Generate biquaternion theta features for time series.
    
    Represents each coefficient as: θ_k = a_k + i*b_k + j*c_k + k*d_k
    where tau = t + j*psi (biquaternionic time)
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    q : float
        Modular parameter
    n_terms : int
        Number of theta series terms
    n_freqs : int
        Number of frequencies
    psi : float
        Imaginary quaternionic phase component
        
    Returns
    -------
    features : np.ndarray
        Feature matrix with biquaternion components, shape (n_samples, n_features)
    """
    t = np.arange(n_samples)
    t_norm = t / n_samples  # Normalize to [0, 1]
    
    features = []
    
    # Generate features at different frequencies and phases
    for k in range(n_freqs):
        omega = 0.5 + k * 0.3  # Frequencies
        
        # For each term, generate biquaternion components
        for n in range(-n_terms // 2, n_terms // 2 + 1):
            if n == 0:
                continue
            
            # Real time component
            phase_t = np.pi * n**2 * q * t_norm + 2 * np.pi * n * omega * t_norm
            
            # Quaternionic imaginary component (psi)
            phase_psi = np.pi * n**2 * q * psi
            
            # Biquaternion components: a + i*b + j*c + k*d
            # a_k (real part)
            features.append(np.cos(phase_t) * np.cosh(phase_psi))
            # b_k (i component)
            features.append(np.sin(phase_t) * np.cosh(phase_psi))
            # c_k (j component)
            features.append(np.cos(phase_t) * np.sinh(phase_psi))
            # d_k (k component)
            features.append(np.sin(phase_t) * np.sinh(phase_psi))
    
    return np.column_stack(features)


def generate_theta_features_1d(n_samples, q=0.5, n_terms=16, n_freqs=8):
    """
    Generate simplified 1D theta-like features for time series.
    
    This creates features suitable for walk-forward prediction on 1D price series.
    Each feature is a basis function evaluated at each time point.
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    q : float
        Modular parameter
    n_terms : int
        Number of theta series terms
    n_freqs : int
        Number of frequencies
        
    Returns
    -------
    features : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    """
    t = np.arange(n_samples)
    t_norm = t / n_samples  # Normalize to [0, 1]
    
    features = []
    
    # Generate features at different frequencies and phases
    for k in range(n_freqs):
        omega = 0.5 + k * 0.3  # Frequencies
        
        # Real and imaginary parts of theta-like functions
        for n in range(-n_terms // 2, n_terms // 2 + 1):
            if n == 0:
                continue
            # Theta-like basis: e^(i*pi*n^2*q*t) * e^(2*pi*i*n*omega*t)
            phase = np.pi * n**2 * q * t_norm + 2 * np.pi * n * omega * t_norm
            features.append(np.cos(phase))
            features.append(np.sin(phase))
    
    return np.column_stack(features)


# ============================================================================
# Drift Term Support (v9)
# ============================================================================

def compute_drift_term(prices, beta0=0.0, beta1=0.0, ema_span=16):
    """
    Compute Fokker-Planck drift term capturing macro bias.
    
    A_t = β₀ + β₁ * tanh(EMA₁₆(r_t))
    
    Parameters
    ----------
    prices : np.ndarray
        Price time series
    beta0 : float
        Drift baseline parameter
    beta1 : float
        Drift sensitivity parameter
    ema_span : int
        EMA span for drift computation
        
    Returns
    -------
    drift : np.ndarray
        Drift term values
    """
    if len(prices) < 2:
        return np.zeros(len(prices))
    
    # Compute returns
    returns = np.diff(prices)
    returns = np.concatenate([[0], returns])  # Prepend 0 to match length
    
    # Compute EMA of returns using pandas for convenience
    ema_drift = pd.Series(returns).ewm(span=ema_span, adjust=False).mean().values
    
    # Apply drift formula
    drift = beta0 + beta1 * np.tanh(ema_drift)
    
    return drift


def fit_drift_parameters(prices, predictions, ema_span=16):
    """
    Fit drift parameters β₀ and β₁ using least squares.
    
    Parameters
    ----------
    prices : np.ndarray
        Price time series
    predictions : np.ndarray
        Base predictions without drift
    ema_span : int
        EMA span for drift computation
        
    Returns
    -------
    beta0, beta1 : tuple of float
        Fitted drift parameters
    """
    if len(prices) < 2 or len(predictions) < 2:
        return 0.0, 0.0
    
    # Compute returns and EMA
    returns = np.diff(prices)
    returns = np.concatenate([[0], returns])
    ema_drift = pd.Series(returns).ewm(span=ema_span, adjust=False).mean().values
    
    # Match lengths
    min_len = min(len(ema_drift), len(predictions))
    ema_drift = ema_drift[:min_len]
    
    # Design matrix for [β₀, β₁]
    X = np.column_stack([
        np.ones(min_len),
        np.tanh(ema_drift)
    ])
    
    # Residuals (what drift should explain)
    if len(prices) >= min_len + 1:
        actuals = np.diff(prices)[:min_len]
        y_residual = actuals[:min_len] - predictions[:min_len]
        
        # Solve least squares
        try:
            params = np.linalg.lstsq(X, y_residual, rcond=None)[0]
            return float(params[0]), float(params[1])
        except:
            return 0.0, 0.0
    
    return 0.0, 0.0


# ============================================================================
# PCA Regime Detection (v9)
# ============================================================================

def detect_regimes_pca(theta_features, n_components=2, n_clusters=2):
    """
    Detect market regimes using PCA on theta features.
    
    Parameters
    ----------
    theta_features : np.ndarray
        Theta feature matrix, shape (n_samples, n_features)
    n_components : int
        Number of PCA components
    n_clusters : int
        Number of regime clusters
        
    Returns
    -------
    pca_coords : np.ndarray
        PCA coordinates, shape (n_samples, n_components)
    regimes : np.ndarray
        Regime labels, shape (n_samples,)
    pca_model : PCA
        Fitted PCA model
    """
    if len(theta_features) < n_clusters:
        # Not enough samples, return dummy values
        return np.zeros((len(theta_features), n_components)), \
               np.zeros(len(theta_features), dtype=int), \
               None
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(theta_features)
    
    # Cluster in PCA space
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(pca_coords)
    except:
        regimes = np.zeros(len(theta_features), dtype=int)
    
    return pca_coords, regimes, pca


def walk_forward_predict(prices, window, horizon, q=0.5, n_terms=16, n_freqs=8, 
                         ridge_lambda=1.0, enable_biquaternion=False, 
                         enable_drift=False, drift_beta0=0.0, drift_beta1=0.0,
                         enable_pca_regimes=False):
    """
    Walk-forward prediction with theta basis.
    
    Version 9 enhancements:
    - Optional biquaternionic time support
    - Optional Fokker-Planck drift term
    - Optional PCA-based regime detection
    
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
        Ridge regularization parameter
    enable_biquaternion : bool
        Use biquaternionic features
    enable_drift : bool
        Add Fokker-Planck drift term
    drift_beta0 : float
        Drift baseline (used if enable_drift=False, else fitted)
    drift_beta1 : float
        Drift sensitivity (used if enable_drift=False, else fitted)
    enable_pca_regimes : bool
        Enable PCA regime detection
        
    Returns
    -------
    predictions : dict
        Dictionary containing predictions and actuals
    """
    n = len(prices)
    
    if n < window + horizon:
        raise ValueError(f"Need at least {window + horizon} samples, got {n}")
    
    # Compute price deltas (returns)
    deltas = np.diff(prices)
    
    predictions = []
    predictions_base = []  # Without drift
    actuals = []
    timestamps = []
    drift_values = []
    regime_labels = []
    pca_coords_list = []
    
    # Walk forward through the data
    for t in range(window, n - horizon):
        # Training window: [t-window : t)
        train_prices = prices[t - window : t]
        
        # Generate features for training window
        if enable_biquaternion:
            X_train = generate_theta_features_biquat(window, q=q, n_terms=n_terms, n_freqs=n_freqs)
        else:
            X_train = generate_theta_features_1d(window, q=q, n_terms=n_terms, n_freqs=n_freqs)
        
        # y_train: predict deltas within the window
        # Use features at time i to predict price[i+1] - price[i]
        window_deltas = np.diff(train_prices)  # Length: window - 1
        
        # Use only the first window-1 features to match
        X_train = X_train[:-1, :]  # Now has window-1 rows
        y_train = window_deltas
        
        # Standardize features
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_std[X_std == 0] = 1.0  # Avoid division by zero
        X_train_std = (X_train - X_mean) / X_std
        
        # Ridge regression: beta = (X^T X + lambda*I)^{-1} X^T y
        n_features = X_train_std.shape[1]
        try:
            XtX = X_train_std.T @ X_train_std
            reg_matrix = ridge_lambda * np.eye(n_features)
            beta = np.linalg.solve(XtX + reg_matrix, X_train_std.T @ y_train)
        except np.linalg.LinAlgError:
            # If singular, skip this prediction
            continue
        
        # Generate features for prediction point (at time t)
        # Use the last feature from the training window
        if enable_biquaternion:
            X_pred_full = generate_theta_features_biquat(window, q=q, n_terms=n_terms, n_freqs=n_freqs)
        else:
            X_pred_full = generate_theta_features_1d(window, q=q, n_terms=n_terms, n_freqs=n_freqs)
        X_pred = X_pred_full[-1, :].reshape(1, -1)
        
        # Standardize using training statistics
        X_pred_std = (X_pred - X_mean) / X_std
        
        # Predict delta at t (which corresponds to price change from t to t+1)
        # To predict at t+horizon, we make a simple projection
        delta_pred = X_pred_std @ beta
        
        # For horizon > 1, accumulate predictions (simplified approach)
        # A more sophisticated approach would recursively predict
        pred_value_base = delta_pred[0] * horizon  # Scale by horizon
        
        # PCA regime detection
        regime = 0
        pca_coord = np.zeros(2)
        if enable_pca_regimes and len(X_train_std) > 10:
            pca_coords, regimes, _ = detect_regimes_pca(X_train_std, n_components=2, n_clusters=2)
            regime = regimes[-1]  # Use most recent regime
            pca_coord = pca_coords[-1]
        
        # Drift term
        drift = 0.0
        if enable_drift:
            # Fit drift parameters on training window
            train_preds = X_train_std @ beta
            beta0_fit, beta1_fit = fit_drift_parameters(train_prices, train_preds)
            
            # Compute drift at prediction point
            drift_at_t = compute_drift_term(prices[t-window:t+1], beta0_fit, beta1_fit)
            drift = drift_at_t[-1] * horizon
            
            # Adaptive drift based on regime
            if enable_pca_regimes:
                # Scale drift by regime (regime 0 = trend, regime 1 = mean-reverting)
                drift_scale = 1.0 if regime == 0 else 0.5
                drift *= drift_scale
        
        pred_value = pred_value_base + drift
        
        # Actual future delta
        actual_delta = prices[t + horizon] - prices[t]
        
        predictions.append(pred_value)
        predictions_base.append(pred_value_base)
        actuals.append(actual_delta)
        timestamps.append(t)
        drift_values.append(drift)
        regime_labels.append(regime)
        pca_coords_list.append(pca_coord)
    
    return {
        'predictions': np.array(predictions),
        'predictions_base': np.array(predictions_base),
        'actuals': np.array(actuals),
        'timestamps': np.array(timestamps),
        'drift_values': np.array(drift_values),
        'regime_labels': np.array(regime_labels),
        'pca_coords': np.array(pca_coords_list)
    }


def compute_metrics(predictions, actuals):
    """
    Compute prediction metrics.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    actuals : np.ndarray
        Actual values
        
    Returns
    -------
    metrics : dict
        Dictionary of metrics
    """
    n = len(predictions)
    
    if n == 0:
        return {
            'n_samples': 0,
            'correlation': np.nan,
            'correlation_pvalue': np.nan,
            'hit_rate': np.nan,
            'mae': np.nan,
            'rmse': np.nan,
            'sharpe_ratio': np.nan,
            'cumulative_pnl': np.nan
        }
    
    # Correlation
    if n > 2:
        corr, p_val = pearsonr(predictions, actuals)
    else:
        corr, p_val = np.nan, np.nan
    
    # Directional accuracy (hit rate)
    pred_signs = np.sign(predictions)
    actual_signs = np.sign(actuals)
    hit_rate = np.mean(pred_signs == actual_signs)
    
    # MAE and RMSE
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    # Trading simulation (simple long/short based on prediction sign)
    # Assume we go long if prediction is positive, short if negative
    # PnL = sign(prediction) * actual_return
    position_pnl = pred_signs * actuals
    
    # Apply transaction cost (e.g., 0.1% per trade)
    transaction_cost = 0.001
    
    # Count position changes
    position_changes = np.sum(np.abs(np.diff(np.concatenate([[0], pred_signs]))))
    total_transaction_cost = position_changes * transaction_cost * np.mean(np.abs(actuals))
    
    # Cumulative PnL
    cumulative_pnl = np.sum(position_pnl) - total_transaction_cost
    
    # Sharpe ratio (annualized, assuming daily data)
    if np.std(position_pnl) > 0:
        sharpe_ratio = np.mean(position_pnl) / np.std(position_pnl) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan
    
    metrics = {
        'n_samples': int(n),
        'correlation': float(corr),
        'correlation_pvalue': float(p_val),
        'hit_rate': float(hit_rate),
        'mae': float(mae),
        'rmse': float(rmse),
        'sharpe_ratio': float(sharpe_ratio),
        'cumulative_pnl': float(cumulative_pnl),
        'total_transaction_cost': float(total_transaction_cost),
        'n_trades': int(position_changes)
    }
    
    return metrics


def plot_predictions(results_by_horizon, outdir):
    """
    Plot predictions vs actuals for each horizon.
    """
    n_horizons = len(results_by_horizon)
    
    fig, axes = plt.subplots(n_horizons, 1, figsize=(12, 4 * n_horizons))
    if n_horizons == 1:
        axes = [axes]
    
    for idx, (horizon, result) in enumerate(sorted(results_by_horizon.items())):
        ax = axes[idx]
        
        preds = result['predictions']
        actuals = result['actuals']
        
        # Scatter plot
        ax.scatter(actuals, preds, alpha=0.5, s=10)
        
        # Diagonal line (perfect prediction)
        min_val = min(actuals.min(), preds.min())
        max_val = max(actuals.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect prediction')
        
        # Metrics
        metrics = result['metrics']
        ax.set_xlabel('Actual Delta')
        ax.set_ylabel('Predicted Delta')
        ax.set_title(f'Horizon {horizon} | r={metrics["correlation"]:.3f}, hit_rate={metrics["hit_rate"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'theta_predictions.png'), dpi=150)
    plt.close()
    print(f"Saved predictions plot to {outdir}/theta_predictions.png")


def plot_cumulative_pnl(results_by_horizon, outdir):
    """
    Plot cumulative PnL over time for each horizon.
    """
    fig, axes = plt.subplots(len(results_by_horizon), 1, 
                            figsize=(12, 4 * len(results_by_horizon)))
    if len(results_by_horizon) == 1:
        axes = [axes]
    
    for idx, (horizon, result) in enumerate(sorted(results_by_horizon.items())):
        ax = axes[idx]
        
        preds = result['predictions']
        actuals = result['actuals']
        
        # Calculate cumulative PnL
        pred_signs = np.sign(preds)
        position_pnl = pred_signs * actuals
        cumulative = np.cumsum(position_pnl)
        
        ax.plot(cumulative, linewidth=1)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative PnL')
        ax.set_title(f'Horizon {horizon} | Total PnL={result["metrics"]["cumulative_pnl"]:.4f}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'theta_trade_sim.png'), dpi=150)
    plt.close()
    print(f"Saved trading simulation plot to {outdir}/theta_trade_sim.png")


# ============================================================================
# v9 Visualization Functions
# ============================================================================

def plot_drift_overlay(results_by_horizon, outdir):
    """
    Plot drift term overlay on predictions (v9).
    """
    fig, axes = plt.subplots(len(results_by_horizon), 1, 
                            figsize=(12, 4 * len(results_by_horizon)))
    if len(results_by_horizon) == 1:
        axes = [axes]
    
    for idx, (horizon, result) in enumerate(sorted(results_by_horizon.items())):
        ax = axes[idx]
        
        if 'predictions_base' in result and 'drift_values' in result:
            preds_base = result['predictions_base']
            drift = result['drift_values']
            preds_full = result['predictions']
            timestamps = result['timestamps']
            
            ax.plot(timestamps, preds_base, label='Base Prediction', alpha=0.7, linewidth=1)
            ax.plot(timestamps, drift, label='Drift Term', alpha=0.7, linewidth=1)
            ax.plot(timestamps, preds_full, label='Full Prediction (Base + Drift)', 
                   alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Predicted Delta')
            ax.set_title(f'Horizon {horizon} - Drift Overlay')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Drift not enabled', ha='center', va='center')
            ax.set_title(f'Horizon {horizon} - Drift Overlay (disabled)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'theta_drift_overlay.png'), dpi=150)
    plt.close()
    print(f"Saved drift overlay plot to {outdir}/theta_drift_overlay.png")


def plot_regime_clusters(results_by_horizon, outdir):
    """
    Plot PCA regime clusters (v9).
    """
    fig, axes = plt.subplots(len(results_by_horizon), 1, 
                            figsize=(12, 6 * len(results_by_horizon)))
    if len(results_by_horizon) == 1:
        axes = [axes]
    
    for idx, (horizon, result) in enumerate(sorted(results_by_horizon.items())):
        ax = axes[idx]
        
        if 'pca_coords' in result and 'regime_labels' in result:
            pca_coords = result['pca_coords']
            regimes = result['regime_labels']
            
            if len(pca_coords) > 0 and pca_coords.shape[1] == 2:
                # Scatter plot colored by regime
                scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                                   c=regimes, cmap='viridis', alpha=0.6, s=20)
                plt.colorbar(scatter, ax=ax, label='Regime')
                
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
                ax.set_title(f'Horizon {horizon} - PCA Regime Clusters')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient PCA data', ha='center', va='center')
                ax.set_title(f'Horizon {horizon} - PCA Regime Clusters (no data)')
        else:
            ax.text(0.5, 0.5, 'PCA regimes not enabled', ha='center', va='center')
            ax.set_title(f'Horizon {horizon} - PCA Regime Clusters (disabled)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'theta_regime_clusters.png'), dpi=150)
    plt.close()
    print(f"Saved regime clusters plot to {outdir}/theta_regime_clusters.png")


def plot_biquat_projection(results_by_horizon, outdir):
    """
    Plot biquaternion projection visualization (v9).
    """
    fig, axes = plt.subplots(len(results_by_horizon), 1, 
                            figsize=(12, 4 * len(results_by_horizon)))
    if len(results_by_horizon) == 1:
        axes = [axes]
    
    for idx, (horizon, result) in enumerate(sorted(results_by_horizon.items())):
        ax = axes[idx]
        
        preds = result['predictions']
        actuals = result['actuals']
        timestamps = result['timestamps']
        
        # Show prediction evolution over time
        ax.plot(timestamps, preds, label='Predictions', alpha=0.7, linewidth=1)
        ax.plot(timestamps, actuals, label='Actuals', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Delta')
        ax.set_title(f'Horizon {horizon} - Biquaternion Projection Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'theta_biquat_projection.png'), dpi=150)
    plt.close()
    print(f"Saved biquaternion projection plot to {outdir}/theta_biquat_projection.png")


def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward prediction with theta basis (v9 with biquaternion, drift, and PCA)'
    )
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV with price data')
    parser.add_argument('--price-col', type=str, default='close',
                       help='Column name for price (default: close)')
    parser.add_argument('--window', type=int, default=512,
                       help='Training window size (default: 512)')
    parser.add_argument('--horizons', type=int, nargs='+', 
                       default=[1, 2, 4, 8, 16, 32],
                       help='Prediction horizons to test (default: 1 2 4 8 16 32)')
    parser.add_argument('--q', type=float, default=0.5,
                       help='Modular parameter (default: 0.5)')
    parser.add_argument('--n-terms', type=int, default=16,
                       help='Number of theta terms (default: 16)')
    parser.add_argument('--n-freqs', type=int, default=8,
                       help='Number of frequencies (default: 8)')
    parser.add_argument('--ridge-lambda', type=float, default=1.0,
                       help='Ridge regularization parameter (default: 1.0)')
    
    # v9 new parameters
    parser.add_argument('--enable-biquaternion', action='store_true',
                       help='Enable biquaternionic time support (v9)')
    parser.add_argument('--enable-drift', action='store_true',
                       help='Enable Fokker-Planck drift term (v9)')
    parser.add_argument('--drift-beta0', type=float, default=0.0,
                       help='Drift baseline parameter (default: 0.0)')
    parser.add_argument('--drift-beta1', type=float, default=0.0,
                       help='Drift sensitivity parameter (default: 0.0)')
    parser.add_argument('--enable-pca-regimes', action='store_true',
                       help='Enable PCA regime detection (v9)')
    
    parser.add_argument('--outdir', type=str, default='theta_output',
                       help='Output directory (default: theta_output)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 60)
    print("Theta Walk-Forward Prediction v9")
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
    print(f"\nv9 Features:")
    print(f"  Biquaternion: {args.enable_biquaternion}")
    print(f"  Drift: {args.enable_drift}")
    if args.enable_drift:
        print(f"    beta0: {args.drift_beta0}, beta1: {args.drift_beta1}")
    print(f"  PCA Regimes: {args.enable_pca_regimes}")
    print()
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if args.price_col not in df.columns:
        raise ValueError(f"Column '{args.price_col}' not found. Available: {list(df.columns)}")
    
    prices = df[args.price_col].values
    print(f"Loaded {len(prices)} price samples")
    print(f"Price range: [{prices.min():.2f}, {prices.max():.2f}]")
    print()
    
    # Run predictions for each horizon
    results_by_horizon = {}
    all_metrics = []
    
    for horizon in args.horizons:
        print(f"Testing horizon {horizon}...")
        
        try:
            result = walk_forward_predict(
                prices=prices,
                window=args.window,
                horizon=horizon,
                q=args.q,
                n_terms=args.n_terms,
                n_freqs=args.n_freqs,
                ridge_lambda=args.ridge_lambda,
                enable_biquaternion=args.enable_biquaternion,
                enable_drift=args.enable_drift,
                drift_beta0=args.drift_beta0,
                drift_beta1=args.drift_beta1,
                enable_pca_regimes=args.enable_pca_regimes
            )
            
            metrics = compute_metrics(result['predictions'], result['actuals'])
            result['metrics'] = metrics
            
            results_by_horizon[horizon] = result
            
            # Add horizon to metrics for summary
            metrics['horizon'] = horizon
            all_metrics.append(metrics)
            
            print(f"  n_samples: {metrics['n_samples']}")
            print(f"  correlation: {metrics['correlation']:.4f}")
            print(f"  hit_rate: {metrics['hit_rate']:.4f}")
            print(f"  sharpe_ratio: {metrics['sharpe_ratio']:.4f}")
            print(f"  cumulative_pnl: {metrics['cumulative_pnl']:.4f}")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    # Save results
    print("Saving outputs...")
    
    # Save summary metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(args.outdir, 'theta_prediction_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")
    
    # Save detailed predictions for each horizon
    for horizon, result in results_by_horizon.items():
        pred_df = pd.DataFrame({
            'timestamp': result['timestamps'],
            'prediction': result['predictions'],
            'actual': result['actuals']
        })
        # Add v9 fields if available
        if 'drift_values' in result and len(result['drift_values']) > 0:
            pred_df['drift'] = result['drift_values']
        if 'regime_labels' in result and len(result['regime_labels']) > 0:
            pred_df['regime'] = result['regime_labels']
        
        pred_path = os.path.join(args.outdir, f'theta_predictions_h{horizon}.csv')
        pred_df.to_csv(pred_path, index=False)
    print(f"Saved detailed predictions for each horizon")
    
    # Generate plots
    print("\nGenerating plots...")
    if results_by_horizon:
        plot_predictions(results_by_horizon, args.outdir)
        plot_cumulative_pnl(results_by_horizon, args.outdir)
        
        # v9 plots
        if args.enable_biquaternion:
            plot_biquat_projection(results_by_horizon, args.outdir)
        if args.enable_drift:
            plot_drift_overlay(results_by_horizon, args.outdir)
        if args.enable_pca_regimes:
            plot_regime_clusters(results_by_horizon, args.outdir)
    
    print("\n" + "=" * 60)
    print("Prediction complete!")
    print("=" * 60)
    
    # Print summary
    if all_metrics:
        avg_corr = np.mean([m['correlation'] for m in all_metrics if not np.isnan(m['correlation'])])
        avg_hit = np.mean([m['hit_rate'] for m in all_metrics if not np.isnan(m['hit_rate'])])
        print(f"\nAverage correlation: {avg_corr:.4f}")
        print(f"Average hit rate: {avg_hit:.4f}")
        
        # Find best horizon
        best_corr_idx = np.argmax([m['correlation'] for m in all_metrics])
        best_horizon = all_metrics[best_corr_idx]['horizon']
        best_corr = all_metrics[best_corr_idx]['correlation']
        print(f"Best horizon (by correlation): {best_horizon} (r={best_corr:.4f})")


if __name__ == '__main__':
    main()
