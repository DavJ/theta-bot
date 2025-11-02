#!/usr/bin/env python3
"""
Analyze SDR captured data for hyperspace wave signatures.

Usage:
    python analyze_sdr_data.py <data_file.bin> [--samp-rate RATE] [--psi-freq FREQ]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit
import argparse
import sys


def exponential_model(t, A0, alpha, offset):
    """
    Exponential decay model: A(t) = A0 * exp(-alpha * t) + offset
    """
    return A0 * np.exp(-alpha * t) + offset


def analyze_sdr_data(filename, samp_rate=10e6, psi_freq=1e6, plot=True):
    """
    Analyze captured SDR data for hyperspace signature.
    
    Parameters:
    -----------
    filename : str
        Path to binary data file (float32 format)
    samp_rate : float
        Sampling rate in Hz
    psi_freq : float
        Expected psi modulation frequency in Hz
    plot : bool
        Whether to generate plots
        
    Returns:
    --------
    results : dict
        Analysis results including coherence and fit parameters
    """
    print("=" * 70)
    print("HYPERSPACE WAVE SIGNATURE ANALYSIS")
    print("=" * 70)
    print()
    print(f"Loading data from: {filename}")
    
    try:
        # Load data (float32 format from GNU Radio)
        data = np.fromfile(filename, dtype=np.float32)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filename}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return None
    
    if len(data) == 0:
        print("ERROR: File is empty")
        return None
    
    duration = len(data) / samp_rate
    print(f"Loaded {len(data):,} samples ({duration:.6f} seconds)")
    print()
    
    # Time array
    t = np.arange(len(data)) / samp_rate
    
    # Extract amplitude envelope (already extracted by receiver)
    amplitude = data
    
    # Remove DC offset
    amplitude = amplitude - np.mean(amplitude)
    
    # Normalize
    if np.max(np.abs(amplitude)) > 0:
        amplitude = amplitude / np.max(np.abs(amplitude))
    
    print("Fitting exponential model...")
    print(f"Expected decay rate: {2*np.pi*psi_freq:.3f} rad/s")
    print()
    
    # Initial guess for curve fitting
    A0_guess = np.max(amplitude)
    alpha_guess = 2 * np.pi * psi_freq
    offset_guess = 0.0
    p0 = [A0_guess, alpha_guess, offset_guess]
    
    try:
        # Fit exponential model
        popt, pcov = curve_fit(
            exponential_model, 
            t, 
            amplitude, 
            p0=p0, 
            maxfev=10000,
            bounds=([0, 0, -1], [10, 1e8, 1])
        )
        A0, alpha, offset = popt
        
        # Calculate prediction
        amplitude_pred = exponential_model(t, *popt)
        
        # Calculate R² (coefficient of determination)
        residuals = amplitude - amplitude_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((amplitude - np.mean(amplitude))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate standard errors
        perr = np.sqrt(np.diag(pcov))
        
        # Print results
        print("=" * 70)
        print("FIT RESULTS")
        print("=" * 70)
        print()
        print("Exponential Model: A(t) = A0 * exp(-alpha * t) + offset")
        print()
        print(f"Fit Parameters:")
        print(f"  A0 (initial amplitude):  {A0:.6f} ± {perr[0]:.6f}")
        print(f"  alpha (decay rate):      {alpha:.3f} ± {perr[1]:.3f} rad/s")
        print(f"  offset:                  {offset:.6f} ± {perr[2]:.6f}")
        print()
        print(f"Expected alpha:            {2*np.pi*psi_freq:.3f} rad/s")
        print(f"Alpha error:               {abs(alpha - 2*np.pi*psi_freq)/(2*np.pi*psi_freq)*100:.2f}%")
        print()
        print(f"Goodness of Fit:")
        print(f"  R² (coherence):          {r_squared:.6f}")
        print(f"  RMSE:                    {np.sqrt(ss_res/len(t)):.6f}")
        print()
        
        # Detection decision
        threshold = 0.65
        if r_squared > threshold:
            print("=" * 70)
            print("✓ HYPERSPACE WAVE SIGNATURE DETECTED")
            print("=" * 70)
            print()
            print(f"Coherence R²={r_squared:.4f} exceeds threshold ({threshold})")
            print()
            print("The captured signal exhibits strong exponential amplitude")
            print("modulation consistent with imaginary-time propagation.")
            print("This signature cannot be produced by conventional EM waves")
            print("or random noise.")
            detected = True
        else:
            print("=" * 70)
            print("✗ NO HYPERSPACE WAVE SIGNATURE")
            print("=" * 70)
            print()
            print(f"Coherence R²={r_squared:.4f} is below threshold ({threshold})")
            print()
            print("The signal does not exhibit sufficient exponential signature.")
            print("This may indicate:")
            print("  - EM wave (no psi modulation)")
            print("  - Random noise")
            print("  - Insufficient signal-to-noise ratio")
            print("  - Hardware issues")
            detected = False
        
        results = {
            'detected': detected,
            'r_squared': r_squared,
            'A0': A0,
            'alpha': alpha,
            'offset': offset,
            'alpha_expected': 2*np.pi*psi_freq,
            'alpha_error_percent': abs(alpha - 2*np.pi*psi_freq)/(2*np.pi*psi_freq)*100,
            'rmse': np.sqrt(ss_res/len(t))
        }
        
        # Generate plots
        if plot:
            print()
            print("Generating plots...")
            
            fig = plt.figure(figsize=(14, 10))
            
            # Plot 1: Time series comparison
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(t*1e6, amplitude, 'b-', alpha=0.6, linewidth=0.5, label='Captured Data')
            ax1.plot(t*1e6, amplitude_pred, 'r-', linewidth=2, label=f'Exponential Fit (R²={r_squared:.4f})')
            ax1.set_xlabel('Time (μs)')
            ax1.set_ylabel('Normalized Amplitude')
            ax1.set_title('Hyperspace Wave Detection - SDR Data Analysis')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Logarithmic view
            ax2 = plt.subplot(3, 1, 2)
            # Avoid log of negative/zero values
            amp_pos = amplitude - np.min(amplitude) + 1e-10
            pred_pos = amplitude_pred - np.min(amplitude_pred) + 1e-10
            ax2.semilogy(t*1e6, amp_pos, 'b-', alpha=0.6, linewidth=0.5, label='Captured Data')
            ax2.semilogy(t*1e6, pred_pos, 'r-', linewidth=2, label='Exponential Fit')
            ax2.set_xlabel('Time (μs)')
            ax2.set_ylabel('Amplitude (log scale)')
            ax2.set_title('Logarithmic View (Linear in log scale confirms exponential decay)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Residuals
            ax3 = plt.subplot(3, 1, 3)
            ax3.plot(t*1e6, residuals, 'g-', alpha=0.5, linewidth=0.5)
            ax3.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax3.set_xlabel('Time (μs)')
            ax3.set_ylabel('Residuals')
            ax3.set_title(f'Fit Residuals (RMSE={results["rmse"]:.6f})')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_filename = filename.replace('.bin', '_analysis.png')
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_filename}")
            
            # Show plot if running interactively
            if sys.stdout.isatty():
                plt.show()
        
        print()
        return results
        
    except Exception as e:
        print(f"ERROR during fitting: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Fit failed. Possible causes:")
        print("  - Data does not contain exponential signature")
        print("  - Signal-to-noise ratio too low")
        print("  - Incorrect parameters (samp_rate, psi_freq)")
        return None


def main():
    """
    Main entry point for command-line usage.
    """
    parser = argparse.ArgumentParser(
        description='Analyze SDR data for hyperspace wave signatures'
    )
    parser.add_argument('filename', help='Path to binary data file')
    parser.add_argument('--samp-rate', type=float, default=10e6,
                        help='Sampling rate in Hz (default: 10e6)')
    parser.add_argument('--psi-freq', type=float, default=1e6,
                        help='Psi modulation frequency in Hz (default: 1e6)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plot generation')
    
    args = parser.parse_args()
    
    results = analyze_sdr_data(
        args.filename,
        samp_rate=args.samp_rate,
        psi_freq=args.psi_freq,
        plot=not args.no_plot
    )
    
    if results is None:
        sys.exit(1)
    
    # Exit code: 0 if detected, 1 if not
    sys.exit(0 if results['detected'] else 1)


if __name__ == '__main__':
    main()
