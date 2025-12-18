#!/usr/bin/env python
"""
Example script demonstrating dual-stream theta + Mellin model usage.

This script shows how to:
1. Load OHLCV data
2. Build dual-stream features (theta + Mellin)
3. Train and evaluate the DualStreamModel
4. Run walk-forward validation with backtesting

Usage:
    python scripts/example_dual_stream.py --data data.csv
    python scripts/example_dual_stream.py --config configs/dual_stream_example.yaml
"""

import argparse
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from theta_bot_averaging.features import build_dual_stream_inputs
from theta_bot_averaging.validation import run_walkforward


def create_synthetic_data(n_samples: int = 300, output_path: str = None) -> pd.DataFrame:
    """Create synthetic OHLCV data for demonstration."""
    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=n_samples, freq="h")
    t = np.linspace(0, 10 * np.pi, n_samples)
    
    # Price with trend and multiple frequencies
    prices = 100 + 0.5 * t + 20 * np.sin(t) + 8 * np.sin(2.3 * t) + np.random.randn(n_samples)
    volume = 1000 + 400 * np.abs(np.cos(t)) + np.random.rand(n_samples) * 100
    
    df = pd.DataFrame({
        "open": prices + np.random.randn(n_samples) * 0.1,
        "high": prices + np.abs(np.random.randn(n_samples)) * 0.3,
        "low": prices - np.abs(np.random.randn(n_samples)) * 0.2,
        "close": prices,
        "volume": volume,
    }, index=idx)
    
    if output_path:
        df.to_csv(output_path)
        print(f"Synthetic data saved to {output_path}")
    
    return df


def demo_feature_extraction(df: pd.DataFrame):
    """Demonstrate feature extraction."""
    print("\n=== Feature Extraction Demo ===")
    print(f"Input data shape: {df.shape}")
    
    # Build dual-stream inputs
    index, X_theta, X_mellin = build_dual_stream_inputs(
        df,
        window=48,
        q=0.9,
        n_terms=8,
        mellin_k=16,
    )
    
    print(f"Valid samples after feature extraction: {len(index)}")
    print(f"Theta features shape: {X_theta.shape}")
    print(f"Mellin features shape: {X_mellin.shape}")
    print(f"No NaNs: {not np.isnan(X_theta).any() and not np.isnan(X_mellin).any()}")
    
    return index, X_theta, X_mellin


def demo_walkforward(config_path: str):
    """Demonstrate walk-forward validation with dual-stream model."""
    print("\n=== Walk-forward Validation Demo ===")
    print(f"Loading config: {config_path}")
    
    result = run_walkforward(config_path)
    
    print(f"\nResults saved to: {result['output_dir']}")
    print("\nSample metrics:")
    for key, value in list(result['metrics'].items())[:10]:
        print(f"  {key}: {value:.4f}")
    
    # Load predictions
    import glob
    pred_files = glob.glob(f"{result['output_dir']}/predictions.parquet")
    if pred_files:
        preds = pd.read_parquet(pred_files[0])
        print(f"\nPredictions shape: {preds.shape}")
        print(f"Signal distribution:")
        for signal, count in preds['signal'].value_counts().sort_index().items():
            print(f"  {int(signal):+2d}: {count} ({count/len(preds)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Dual-stream model demonstration")
    parser.add_argument("--data", type=str, help="Path to OHLCV CSV file")
    parser.add_argument("--config", type=str, help="Path to walkforward config YAML")
    parser.add_argument("--create-synthetic", action="store_true",
                        help="Create synthetic data for demonstration")
    parser.add_argument("--demo-features", action="store_true",
                        help="Demonstrate feature extraction only")
    
    args = parser.parse_args()
    
    if args.create_synthetic or (not args.data and not args.config):
        print("Creating synthetic data...")
        data_path = "demo_data.csv"
        df = create_synthetic_data(output_path=data_path)
        
        if args.demo_features:
            demo_feature_extraction(df)
            return
        
        # Create demo config
        import yaml
        config_path = "demo_config.yaml"
        cfg = {
            "data_path": data_path,
            "horizon": 1,
            "threshold_bps": 10.0,
            "model_type": "dual_stream",
            "fee_rate": 0.0,
            "slippage_bps": 0.0,
            "spread_bps": 0.0,
            "n_splits": 3,
            "purge": 0,
            "embargo": 0,
            "output_dir": "demo_runs",
            "theta_window": 40,
            "theta_q": 0.9,
            "theta_terms": 6,
            "mellin_k": 12,
            "torch_epochs": 10,
            "torch_batch_size": 16,
            "torch_lr": 1e-3,
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f)
        print(f"Demo config saved to {config_path}")
        args.config = config_path
    
    if args.config:
        demo_walkforward(args.config)
    elif args.data and args.demo_features:
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        demo_feature_extraction(df)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
