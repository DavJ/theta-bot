#!/usr/bin/env python3
"""
Demonstration script showing usage of all Mellin cepstrum modes.
This script creates synthetic data and computes features using all available psi modes.
"""

import numpy as np
import pandas as pd
from spot_bot.features import FeatureConfig, compute_features


def create_synthetic_data(n=500):
    """Create synthetic OHLCV data for demonstration."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 20000 + np.linspace(0, 500, n)
    noise = np.sin(np.linspace(0, 6.28, n)) * 50
    close = base + noise
    
    np.random.seed(42)
    open_noise = np.random.normal(0, 0.0005, size=n)
    spread_noise = np.abs(np.random.normal(0, 0.0005, size=n))
    
    open_ = close * (1 + open_noise)
    high = np.maximum(open_, close) * (1 + spread_noise)
    low = np.minimum(open_, close) * (1 - spread_noise)
    volume = np.full(n, 1.0)
    
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def main():
    print("=" * 80)
    print("Mellin Cepstrum Demonstration")
    print("=" * 80)
    
    # Create synthetic data
    print("\n1. Creating synthetic OHLCV data (500 bars)...")
    ohlcv = create_synthetic_data(500)
    print(f"   Data shape: {ohlcv.shape}")
    print(f"   Date range: {ohlcv.index[0]} to {ohlcv.index[-1]}")
    
    # Test each psi mode
    modes = [
        ("cepstrum", "FFT-based real cepstrum (default)"),
        ("complex_cepstrum", "FFT-based complex cepstrum"),
        ("mellin_cepstrum", "Mellin-based real cepstrum"),
        ("mellin_complex_cepstrum", "Mellin-based complex cepstrum"),
    ]
    
    print("\n2. Computing features with different psi modes...")
    print("-" * 80)
    
    results = {}
    for mode, description in modes:
        print(f"\n   Mode: {mode}")
        print(f"   Description: {description}")
        
        if mode.startswith("mellin"):
            cfg = FeatureConfig(
                psi_mode=mode,
                psi_window=64,
                mellin_grid_n=128,
                mellin_sigma=0.0,
                psi_min_bin=2,
                psi_max_frac=0.25,
            )
        else:
            cfg = FeatureConfig(
                psi_mode=mode,
                psi_window=64,
            )
        
        features = compute_features(ohlcv, cfg)
        results[mode] = features
        
        # Show statistics
        psi_vals = features["psi"].dropna()
        if not psi_vals.empty:
            print(f"   Valid psi values: {len(psi_vals)}")
            print(f"   Psi range: [{psi_vals.min():.4f}, {psi_vals.max():.4f}]")
            print(f"   Psi mean: {psi_vals.mean():.4f}")
            print(f"   Psi std: {psi_vals.std():.4f}")
        else:
            print("   No valid psi values computed")
    
    # Compare modes
    print("\n3. Comparison of last 10 psi values across modes:")
    print("-" * 80)
    comparison = pd.DataFrame({
        mode: results[mode]["psi"].tail(10).values
        for mode, _ in modes
    })
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    
    # Mellin-specific parameter demonstration
    print("\n4. Mellin parameter variations:")
    print("-" * 80)
    
    mellin_configs = [
        ("sigma=0.0", {"mellin_sigma": 0.0}),
        ("sigma=0.5", {"mellin_sigma": 0.5}),
        ("grid_n=64", {"mellin_grid_n": 64}),
        ("grid_n=256", {"mellin_grid_n": 256}),
        ("cmean agg", {"psi_phase_agg": "cmean", "psi_phase_power": 1.5}),
    ]
    
    for label, params in mellin_configs:
        base_params = {
            "psi_mode": "mellin_complex_cepstrum",
            "psi_window": 64,
            "mellin_grid_n": 128,
            "mellin_sigma": 0.0,
            "mellin_detrend_phase": True,
            "psi_min_bin": 2,
            "psi_max_frac": 0.25,
            "psi_phase_agg": "peak",
            "psi_phase_power": 1.0,
        }
        base_params.update(params)
        cfg = FeatureConfig(**base_params)
        features = compute_features(ohlcv, cfg)
        psi_vals = features["psi"].dropna()
        
        if not psi_vals.empty:
            print(f"   {label:20s}: mean={psi_vals.mean():.4f}, std={psi_vals.std():.4f}")
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
