#!/usr/bin/env python3
"""
Example demonstrating quantile-based signal generation.

This example shows how the quantile mode generates signals based on
the distribution of predicted returns rather than fixed thresholds.
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from theta_bot_averaging.utils import generate_signals

def main():
    print("=" * 70)
    print("Quantile vs Threshold Signal Generation - Example")
    print("=" * 70)
    print()
    
    # Generate sample predicted returns
    np.random.seed(42)
    n_samples = 100
    predicted_returns = pd.Series(np.random.randn(n_samples) * 0.01)  # ~1% std
    
    print(f"Generated {n_samples} predicted returns:")
    print(f"  Mean: {predicted_returns.mean():.6f}")
    print(f"  Std:  {predicted_returns.std():.6f}")
    print(f"  Min:  {predicted_returns.min():.6f}")
    print(f"  Max:  {predicted_returns.max():.6f}")
    print()
    
    # Generate signals using threshold mode
    print("-" * 70)
    print("THRESHOLD MODE (Fixed 10 bps = 0.001)")
    print("-" * 70)
    
    threshold_signals = generate_signals(
        predicted_returns,
        mode="threshold",
        positive_threshold=0.001,  # 10 bps
        negative_threshold=-0.001,
    )
    
    long_count = (threshold_signals == 1).sum()
    short_count = (threshold_signals == -1).sum()
    neutral_count = (threshold_signals == 0).sum()
    
    print(f"Signals generated:")
    print(f"  Long (1):    {long_count:3d} ({long_count/n_samples*100:5.1f}%)")
    print(f"  Short (-1):  {short_count:3d} ({short_count/n_samples*100:5.1f}%)")
    print(f"  Neutral (0): {neutral_count:3d} ({neutral_count/n_samples*100:5.1f}%)")
    print()
    
    if long_count > 0:
        print(f"Long signal threshold:  > +0.001 (10 bps)")
        print(f"  Mean predicted return: {predicted_returns[threshold_signals == 1].mean():.6f}")
    if short_count > 0:
        print(f"Short signal threshold: < -0.001 (-10 bps)")
        print(f"  Mean predicted return: {predicted_returns[threshold_signals == -1].mean():.6f}")
    print()
    
    # Generate signals using quantile mode
    print("-" * 70)
    print("QUANTILE MODE (95th/5th percentile)")
    print("-" * 70)
    
    quantile_signals = generate_signals(
        predicted_returns,
        mode="quantile",
        quantile_long=0.95,
        quantile_short=0.05,
    )
    
    long_count = (quantile_signals == 1).sum()
    short_count = (quantile_signals == -1).sum()
    neutral_count = (quantile_signals == 0).sum()
    
    print(f"Signals generated:")
    print(f"  Long (1):    {long_count:3d} ({long_count/n_samples*100:5.1f}%)")
    print(f"  Short (-1):  {short_count:3d} ({short_count/n_samples*100:5.1f}%)")
    print(f"  Neutral (0): {neutral_count:3d} ({neutral_count/n_samples*100:5.1f}%)")
    print()
    
    long_threshold = predicted_returns.quantile(0.95)
    short_threshold = predicted_returns.quantile(0.05)
    
    print(f"Long signal threshold:  > {long_threshold:.6f} (95th percentile)")
    if long_count > 0:
        print(f"  Mean predicted return: {predicted_returns[quantile_signals == 1].mean():.6f}")
    print()
    print(f"Short signal threshold: < {short_threshold:.6f} (5th percentile)")
    if short_count > 0:
        print(f"  Mean predicted return: {predicted_returns[quantile_signals == -1].mean():.6f}")
    print()
    
    # Compare the two approaches
    print("=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    print()
    print("Threshold Mode:")
    print("  ✓ Fixed cutoffs independent of data distribution")
    print("  ✓ Signal count varies with prediction quality")
    print("  ✓ Suitable for production trading with fixed risk parameters")
    print()
    print("Quantile Mode:")
    print("  ✓ Adaptive cutoffs based on prediction distribution")
    print("  ✓ Consistent signal count (~10% total: 5% long, 5% short)")
    print("  ✓ Tests if model can rank opportunities (evaluation only)")
    print("  ✓ Robust to prediction scale/bias issues")
    print()
    print("Use quantile mode to verify the model has ranking ability,")
    print("even if absolute predicted magnitudes are not well calibrated.")
    print()


if __name__ == "__main__":
    main()
