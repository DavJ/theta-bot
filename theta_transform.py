#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_transform.py
------------------
Implement theta transform and inverse transform functions.

The transform projects signals onto the orthonormalized theta basis,
and the inverse reconstructs signals from coefficients.

Functions:
- theta_transform(signal, basis) → coeffs
- theta_inverse(coeffs, basis) → reconstructed

Validation includes:
- Correlation between original and reconstructed
- RMS error
- Energy conservation

Author: Implementation based on COPILOT_BRIEF_v2.md
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def theta_transform(signal, basis):
    """
    Project signal onto orthonormalized theta basis.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (real-valued), shape (n_samples,)
    basis : np.ndarray
        Orthonormalized basis, shape (n_samples, n_basis), complex
        
    Returns
    -------
    coeffs : np.ndarray
        Projection coefficients, shape (n_basis,), complex
    """
    # Project: c_i = <basis_i, signal> = basis_i† · signal
    # In matrix form: coeffs = basis† · signal
    coeffs = basis.conj().T @ signal
    return coeffs


def theta_inverse(coeffs, basis):
    """
    Reconstruct signal from theta coefficients.
    
    Parameters
    ----------
    coeffs : np.ndarray
        Projection coefficients, shape (n_basis,), complex
    basis : np.ndarray
        Orthonormalized basis, shape (n_samples, n_basis), complex
        
    Returns
    -------
    reconstructed : np.ndarray
        Reconstructed signal, shape (n_samples,), real or complex
    """
    # Reconstruct: signal ≈ Σ_i c_i * basis_i = basis @ coeffs
    reconstructed = basis @ coeffs
    return reconstructed


def validate_reconstruction(original, reconstructed):
    """
    Validate reconstruction quality.
    
    Parameters
    ----------
    original : np.ndarray
        Original signal
    reconstructed : np.ndarray
        Reconstructed signal (may be complex)
        
    Returns
    -------
    metrics : dict
        Dictionary containing validation metrics
    """
    # If reconstructed is complex, use real part or magnitude
    if np.iscomplexobj(reconstructed):
        reconstructed_real = np.real(reconstructed)
    else:
        reconstructed_real = reconstructed
    
    # Ensure both are real-valued for correlation
    original_real = np.real(original) if np.iscomplexobj(original) else original
    
    # Correlation
    if len(original_real) > 1:
        corr, p_value = pearsonr(original_real, reconstructed_real)
    else:
        corr, p_value = np.nan, np.nan
    
    # RMS error
    rmse = np.sqrt(np.mean((original_real - reconstructed_real) ** 2))
    
    # Normalized RMS error
    signal_std = np.std(original_real)
    nrmse = rmse / signal_std if signal_std > 0 else np.inf
    
    # Energy conservation
    energy_original = np.sum(original_real ** 2)
    energy_reconstructed = np.sum(reconstructed_real ** 2)
    energy_ratio = energy_reconstructed / energy_original if energy_original > 0 else np.nan
    
    # Mean absolute error
    mae = np.mean(np.abs(original_real - reconstructed_real))
    
    metrics = {
        'correlation': float(corr),
        'correlation_pvalue': float(p_value),
        'rmse': float(rmse),
        'nrmse': float(nrmse),
        'mae': float(mae),
        'energy_original': float(energy_original),
        'energy_reconstructed': float(energy_reconstructed),
        'energy_ratio': float(energy_ratio),
        'signal_mean': float(np.mean(original_real)),
        'signal_std': float(signal_std),
        'n_samples': int(len(original_real))
    }
    
    return metrics


def plot_reconstruction(original, reconstructed, outdir, filename='theta_reconstruction.png'):
    """
    Plot original vs reconstructed signal.
    """
    # If reconstructed is complex, use real part
    if np.iscomplexobj(reconstructed):
        reconstructed_real = np.real(reconstructed)
        has_imag = True
        reconstructed_imag = np.imag(reconstructed)
    else:
        reconstructed_real = reconstructed
        has_imag = False
    
    original_real = np.real(original) if np.iscomplexobj(original) else original
    
    if has_imag:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Real part comparison
        axes[0].plot(original_real, 'b-', alpha=0.7, label='Original', linewidth=1)
        axes[0].plot(reconstructed_real, 'r--', alpha=0.7, label='Reconstructed (Real)', linewidth=1)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Signal Reconstruction - Real Part')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Imaginary part
        axes[1].plot(reconstructed_imag, 'g-', alpha=0.7, label='Reconstructed (Imag)', linewidth=1)
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Reconstructed Signal - Imaginary Part')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Error
        error = original_real - reconstructed_real
        axes[2].plot(error, 'k-', alpha=0.7, linewidth=1)
        axes[2].set_xlabel('Sample')
        axes[2].set_ylabel('Error')
        axes[2].set_title('Reconstruction Error')
        axes[2].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Signal comparison
        axes[0].plot(original_real, 'b-', alpha=0.7, label='Original', linewidth=1)
        axes[0].plot(reconstructed_real, 'r--', alpha=0.7, label='Reconstructed', linewidth=1)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Signal Reconstruction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error
        error = original_real - reconstructed_real
        axes[1].plot(error, 'k-', alpha=0.7, linewidth=1)
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Reconstruction Error')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(outdir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved reconstruction plot to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Test theta transform and inverse transform'
    )
    parser.add_argument('--basis', type=str, required=True,
                       help='Path to theta_basis.npy file')
    parser.add_argument('--signal', type=str, default=None,
                       help='Path to signal file (npy or csv). If not provided, generates test signal.')
    parser.add_argument('--signal-col', type=str, default='close',
                       help='Column name if signal is CSV (default: close)')
    parser.add_argument('--outdir', type=str, default='theta_output',
                       help='Output directory (default: theta_output)')
    parser.add_argument('--test-synthetic', action='store_true',
                       help='Test with synthetic signal (sum of sinusoids)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 60)
    print("Theta Transform Testing")
    print("=" * 60)
    
    # Load basis
    print(f"Loading basis from {args.basis}...")
    basis = np.load(args.basis)
    n_samples, n_basis = basis.shape
    print(f"Basis shape: {basis.shape}")
    print(f"  n_samples: {n_samples}")
    print(f"  n_basis: {n_basis}")
    print()
    
    # Load or generate signal
    if args.signal is not None:
        print(f"Loading signal from {args.signal}...")
        if args.signal.endswith('.npy'):
            signal = np.load(args.signal)
        elif args.signal.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(args.signal)
            if args.signal_col not in df.columns:
                raise ValueError(f"Column '{args.signal_col}' not found in CSV")
            signal = df[args.signal_col].values
        else:
            raise ValueError("Signal file must be .npy or .csv")
        
        # Truncate or pad to match basis size
        if len(signal) > n_samples:
            print(f"Truncating signal from {len(signal)} to {n_samples} samples")
            signal = signal[:n_samples]
        elif len(signal) < n_samples:
            print(f"Padding signal from {len(signal)} to {n_samples} samples")
            signal = np.pad(signal, (0, n_samples - len(signal)), mode='edge')
    else:
        print("Generating synthetic test signal...")
        t = np.linspace(0, 10, n_samples)
        # Sum of sinusoids with different frequencies
        signal = (np.sin(2 * np.pi * 0.5 * t) + 
                 0.5 * np.sin(2 * np.pi * 1.2 * t) + 
                 0.3 * np.sin(2 * np.pi * 2.3 * t) +
                 0.1 * np.random.randn(n_samples))  # Add noise
    
    print(f"Signal shape: {signal.shape}")
    print(f"Signal stats: mean={np.mean(signal):.4f}, std={np.std(signal):.4f}")
    print()
    
    # Perform transform
    print("Performing theta transform...")
    coeffs = theta_transform(signal, basis)
    print(f"Coefficients shape: {coeffs.shape}")
    print(f"Coefficients (first 5): {coeffs[:5]}")
    print()
    
    # Perform inverse transform
    print("Performing inverse theta transform...")
    reconstructed = theta_inverse(coeffs, basis)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print()
    
    # Validate reconstruction
    print("Validating reconstruction...")
    metrics = validate_reconstruction(signal, reconstructed)
    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Save outputs
    print("Saving outputs...")
    
    # Save coefficients
    coeffs_path = os.path.join(args.outdir, 'theta_coeffs.npy')
    np.save(coeffs_path, coeffs)
    print(f"Saved coefficients to {coeffs_path}")
    
    # Save reconstructed signal
    reconstructed_path = os.path.join(args.outdir, 'theta_reconstructed.npy')
    np.save(reconstructed_path, reconstructed)
    print(f"Saved reconstructed signal to {reconstructed_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.outdir, 'theta_transform_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Generate plot
    print("\nGenerating reconstruction plot...")
    plot_reconstruction(signal, reconstructed, args.outdir)
    
    print("\n" + "=" * 60)
    print("Transform testing complete!")
    print("=" * 60)
    
    # Print summary
    if metrics['correlation'] > 0.9:
        print(f"✓ Excellent reconstruction (r = {metrics['correlation']:.4f})")
    elif metrics['correlation'] > 0.7:
        print(f"✓ Good reconstruction (r = {metrics['correlation']:.4f})")
    else:
        print(f"⚠ Warning: Low correlation (r = {metrics['correlation']:.4f})")
    
    if metrics['energy_ratio'] > 0.95 and metrics['energy_ratio'] < 1.05:
        print(f"✓ Energy well conserved (ratio = {metrics['energy_ratio']:.4f})")
    else:
        print(f"⚠ Energy ratio: {metrics['energy_ratio']:.4f}")


if __name__ == '__main__':
    main()
