#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_basis_4d.py
-----------------
Generate a 4D orthonormalized basis using Jacobi theta functions.

This implements the theoretical foundation from CCT/UBT for market time as complex:
    τ = t + iψ
where t is chronological time and ψ is a hidden phase component.

The basis is constructed from:
    Θ(q, τ, φ) = Σ_{n=-N}^{N} e^{iπn²τ} e^{2πinqφ}

The 4D axes are:
- Frequency (ω)
- Phase (φ)
- Imaginary time (ψ)
- Discrete mode (n)

Outputs:
- theta_basis.npy: Generated 4D orthonormal basis
- theta_spectrum.png: Eigenvalue spectra before and after normalization
- theta_projection.png: Toroidal/phase projection visualization
- theta_metrics.json: Orthogonality metrics

Author: Implementation based on COPILOT_BRIEF_v2.md
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr


def jacobi_theta_basis(n, t, q, phi=0.0):
    """
    Compute Jacobi theta basis function for mode n at time t.
    
    Parameters
    ----------
    n : int
        Mode index
    t : float or np.ndarray
        Time coordinate (can be complex: t + i*psi)
    q : float
        Modular parameter (0 < q < 1)
    phi : float or complex
        Phase variable
        
    Returns
    -------
    complex
        Theta basis function value
    """
    return np.exp(1j * np.pi * n**2 * t) * np.exp(2j * np.pi * n * q * phi)


def generate_4d_theta_basis(n_modes=32, n_freqs=8, n_phases=8, n_psi=8, 
                            q=0.5, t_max=1.0):
    """
    Generate 4D orthonormalized theta basis.
    
    Parameters
    ----------
    n_modes : int
        Number of discrete modes n in [-N, N]
    n_freqs : int
        Number of frequency samples (ω axis)
    n_phases : int
        Number of phase samples (φ axis)
    n_psi : int
        Number of imaginary time samples (ψ axis)
    q : float
        Modular parameter
    t_max : float
        Maximum time value
        
    Returns
    -------
    basis : np.ndarray
        Shape (n_samples, n_basis_funcs) where n_samples = n_freqs * n_phases * n_psi
        and n_basis_funcs = (2*n_modes+1)
    coords : dict
        Dictionary containing the coordinate grids
    """
    # Create coordinate grids
    freqs = np.linspace(0.1, 2.0, n_freqs)  # Angular frequencies
    phases = np.linspace(0, 2*np.pi, n_phases, endpoint=False)  # Phase
    psis = np.linspace(0, 1.0, n_psi)  # Imaginary time component
    
    # Mode indices
    n_indices = np.arange(-n_modes, n_modes + 1)
    n_basis = len(n_indices)
    
    # Total number of sample points
    n_samples = n_freqs * n_phases * n_psi
    
    # Initialize basis matrix
    basis = np.zeros((n_samples, n_basis), dtype=np.complex128)
    
    # Generate basis functions at each sample point
    sample_idx = 0
    for omega in freqs:
        for phi in phases:
            for psi in psis:
                # Complex time: τ = t + i*ψ
                for mode_idx, n in enumerate(n_indices):
                    # Sample at time points scaled by frequency
                    t_complex = omega * t_max + 1j * psi
                    basis[sample_idx, mode_idx] = jacobi_theta_basis(n, t_complex, q, phi)
                
                sample_idx += 1
    
    coords = {
        'freqs': freqs,
        'phases': phases,
        'psis': psis,
        'n_indices': n_indices,
        'q': q
    }
    
    return basis, coords


def complex_gram_schmidt(basis):
    """
    Perform complex Gram-Schmidt orthonormalization.
    
    Parameters
    ----------
    basis : np.ndarray
        Complex basis matrix (n_samples, n_basis)
        
    Returns
    -------
    ortho_basis : np.ndarray
        Orthonormalized basis
    overlap_before : np.ndarray
        Overlap matrix before orthonormalization
    overlap_after : np.ndarray
        Overlap matrix after orthonormalization
    """
    n_samples, n_basis = basis.shape
    
    # Compute overlap matrix before orthonormalization
    # O_ij = <Θ_i, Θ_j> = (Θ_i)† · Θ_j / n_samples (normalized inner product)
    overlap_before = (basis.conj().T @ basis) / n_samples
    
    # Use QR decomposition for stable orthonormalization
    # Q is orthonormal in the standard inner product (not normalized)
    Q, R = qr(basis, mode='economic')
    
    # Q from QR is already orthonormal in the sense that Q†Q = I
    # So we use Q directly as the orthonormal basis
    ortho_basis = Q
    
    # Compute overlap matrix after orthonormalization
    # Should be identity matrix
    overlap_after = (ortho_basis.conj().T @ ortho_basis) / n_samples
    
    return ortho_basis, overlap_before, overlap_after


def compute_eigenvalue_spectrum(overlap_matrix):
    """
    Compute eigenvalue spectrum of overlap matrix.
    
    Parameters
    ----------
    overlap_matrix : np.ndarray
        Complex hermitian matrix
        
    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues (real)
    """
    eigenvalues = np.linalg.eigvalsh(overlap_matrix)
    return np.sort(eigenvalues)[::-1]  # Sort descending


def plot_spectrum(eigs_before, eigs_after, outdir):
    """
    Plot eigenvalue spectra before and after orthonormalization.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(eigs_before, 'o-', label='Before orthonormalization')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Spectrum (Before)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(eigs_after, 'o-', label='After orthonormalization', color='green')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Expected (1.0)')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Spectrum (After)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'theta_spectrum.png'), dpi=150)
    plt.close()
    print(f"Saved eigenvalue spectrum to {outdir}/theta_spectrum.png")


def plot_projection(basis, coords, outdir):
    """
    Plot toroidal/phase projection of theta basis.
    """
    n_samples, n_basis = basis.shape
    
    # Take first few modes for visualization
    n_vis_modes = min(4, n_basis)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for mode_idx in range(n_vis_modes):
        ax = axes[mode_idx]
        
        # Reshape basis values to phase x psi grid (averaging over frequencies)
        n_phases = len(coords['phases'])
        n_psi = len(coords['psis'])
        n_freqs = len(coords['freqs'])
        
        # Average over frequencies
        basis_reshaped = basis[:, mode_idx].reshape(n_freqs, n_phases, n_psi)
        basis_avg = np.mean(basis_reshaped, axis=0)  # Average over freqs
        
        # Plot magnitude as phase-space projection
        im = ax.imshow(np.abs(basis_avg), aspect='auto', origin='lower',
                      extent=[0, 1, 0, 2*np.pi], cmap='viridis')
        ax.set_xlabel('ψ (imaginary time)')
        ax.set_ylabel('φ (phase)')
        ax.set_title(f'Mode n={coords["n_indices"][mode_idx]}')
        plt.colorbar(im, ax=ax, label='|Θ|')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'theta_projection.png'), dpi=150)
    plt.close()
    print(f"Saved phase projection to {outdir}/theta_projection.png")


def compute_orthogonality_metrics(overlap_before, overlap_after):
    """
    Compute metrics to validate orthonormalization.
    
    Returns
    -------
    metrics : dict
        Dictionary of orthogonality metrics
    """
    n = overlap_after.shape[0]
    
    # For normalized inner product, identity should be scaled
    # Q†Q = n_samples * I for normalized inner product
    # So we need to check if diagonal elements are constant
    diag_vals = np.diag(overlap_after).real
    expected_diag = diag_vals[0]  # Should all be the same
    
    # Check if Re(O) has constant diagonal
    re_diag_std = np.std(diag_vals)
    
    # Check if Im(O) ≈ 0
    max_im = np.max(np.abs(overlap_after.imag))
    mean_im = np.mean(np.abs(overlap_after.imag))
    
    # Check Hermitian symmetry: O = O†
    hermitian_diff = np.max(np.abs(overlap_after - overlap_after.conj().T))
    
    # Off-diagonal elements before orthonormalization
    mask = ~np.eye(n, dtype=bool)
    off_diag_before = np.abs(overlap_before[mask])
    mean_off_diag_before = np.mean(off_diag_before)
    max_off_diag_before = np.max(off_diag_before)
    
    # Off-diagonal elements after orthonormalization (relative to diagonal)
    off_diag_after = np.abs(overlap_after[mask])
    mean_off_diag_after = np.mean(off_diag_after)
    max_off_diag_after = np.max(off_diag_after)
    relative_off_diag = max_off_diag_after / expected_diag if expected_diag > 0 else 0
    
    metrics = {
        'diagonal_value': float(expected_diag),
        'diagonal_std': float(re_diag_std),
        'max_imaginary_component': float(max_im),
        'mean_imaginary_component': float(mean_im),
        'hermitian_symmetry_error': float(hermitian_diff),
        'off_diagonal_mean_before': float(mean_off_diag_before),
        'off_diagonal_max_before': float(max_off_diag_before),
        'off_diagonal_mean_after': float(mean_off_diag_after),
        'off_diagonal_max_after': float(max_off_diag_after),
        'relative_off_diagonal': float(relative_off_diag),
        'is_orthonormal': bool(re_diag_std < 1e-6 and max_im < 1e-6 and relative_off_diag < 1e-6)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Generate 4D orthonormalized Jacobi theta basis'
    )
    parser.add_argument('--n-modes', type=int, default=16,
                       help='Number of modes: n in [-N, N] (default: 16)')
    parser.add_argument('--n-freqs', type=int, default=8,
                       help='Number of frequency samples (default: 8)')
    parser.add_argument('--n-phases', type=int, default=8,
                       help='Number of phase samples (default: 8)')
    parser.add_argument('--n-psi', type=int, default=8,
                       help='Number of imaginary time samples (default: 8)')
    parser.add_argument('--q', type=float, default=0.5,
                       help='Modular parameter q (0 < q < 1, default: 0.5)')
    parser.add_argument('--outdir', type=str, default='theta_output',
                       help='Output directory (default: theta_output)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    print("=" * 60)
    print("4D Jacobi Theta Basis Generation")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  n_modes: {args.n_modes}")
    print(f"  n_freqs: {args.n_freqs}")
    print(f"  n_phases: {args.n_phases}")
    print(f"  n_psi: {args.n_psi}")
    print(f"  q: {args.q}")
    print(f"  Total basis functions: {2*args.n_modes + 1}")
    print()
    
    # Generate basis
    print("Generating 4D theta basis...")
    basis, coords = generate_4d_theta_basis(
        n_modes=args.n_modes,
        n_freqs=args.n_freqs,
        n_phases=args.n_phases,
        n_psi=args.n_psi,
        q=args.q
    )
    print(f"Basis shape: {basis.shape}")
    print()
    
    # Perform orthonormalization
    print("Performing complex Gram-Schmidt orthonormalization...")
    ortho_basis, overlap_before, overlap_after = complex_gram_schmidt(basis)
    print("Orthonormalization complete.")
    print()
    
    # Compute eigenvalue spectra
    print("Computing eigenvalue spectra...")
    eigs_before = compute_eigenvalue_spectrum(overlap_before)
    eigs_after = compute_eigenvalue_spectrum(overlap_after)
    print(f"Eigenvalues before (first 5): {eigs_before[:5]}")
    print(f"Eigenvalues after (first 5): {eigs_after[:5]}")
    print()
    
    # Compute orthogonality metrics
    print("Computing orthogonality metrics...")
    metrics = compute_orthogonality_metrics(overlap_before, overlap_after)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()
    
    # Save outputs
    print("Saving outputs...")
    
    # Save basis
    basis_path = os.path.join(args.outdir, 'theta_basis.npy')
    np.save(basis_path, ortho_basis)
    print(f"Saved basis to {basis_path}")
    
    # Save coordinates for later use
    coords_path = os.path.join(args.outdir, 'theta_coords.npy')
    np.save(coords_path, coords, allow_pickle=True)
    print(f"Saved coordinates to {coords_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.outdir, 'theta_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Save eigenvalues
    eigs_path = os.path.join(args.outdir, 'theta_eigenvalues.npz')
    np.savez(eigs_path, before=eigs_before, after=eigs_after)
    print(f"Saved eigenvalues to {eigs_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_spectrum(eigs_before, eigs_after, args.outdir)
    plot_projection(ortho_basis, coords, args.outdir)
    
    print("\n" + "=" * 60)
    print("Basis generation complete!")
    print("=" * 60)
    
    # Print validation summary
    if metrics['is_orthonormal']:
        print("✓ Basis is successfully orthonormalized")
        print(f"  Diagonal value: {metrics['diagonal_value']:.6f}")
        print(f"  Relative off-diagonal: {metrics['relative_off_diagonal']:.2e}")
    else:
        print("⚠ Warning: Basis may not be fully orthonormal")
        print(f"  Diagonal std: {metrics['diagonal_std']:.2e}")
        print(f"  Max imaginary component: {metrics['max_imaginary_component']:.2e}")
        print(f"  Relative off-diagonal: {metrics['relative_off_diagonal']:.2e}")


if __name__ == '__main__':
    main()
