#!/usr/bin/env python3
"""
Mellin Cepstrum Parameter Tuning Script

This script provides a reproducible parameter sweep and tuning framework for selecting
optimal parameters for Mellin cepstrum-based regime detection and phase timing.

WORKFLOW:
1. Load OHLCV data from CSV
2. Define parameter grids based on tuning mode (regime, phase, or both)
3. For each parameter configuration:
   - Create FeatureConfig with the parameters
   - Run backtest using run_mean_reversion_backtests
   - Extract metrics from 'regime' (gated) results
   - Compute additional metrics: Sharpe ratio, volatility, max drawdown
4. Save all results to a CSV file
5. Rank configurations by performance (Sharpe desc, max drawdown desc, turnover asc)
6. Print top N configurations

MODES:
- regime: Sweeps parameters for mellin_cepstrum (psi_mode=mellin_cepstrum)
  - Tunable: psi_window, mellin_sigma, mellin_grid_n, psi_min_bin, psi_max_frac
  - Fixed: psi_phase_agg="peak", mellin_eps=default
  
- phase: Sweeps parameters for mellin_complex_cepstrum (psi_mode=mellin_complex_cepstrum)
  - Tunable: mellin_detrend_phase, psi_phase_agg, psi_phase_power, mellin_eps
  - Requires base config (can use results from regime mode)
  
- both: Runs regime sweep first, then uses best configs as base for phase sweep

PARAMETER GRIDS:
The default parameter grids are designed for practical tuning with reasonable runtime.
Users can modify generate_regime_grid() and generate_phase_grid() functions to:
- Add more parameter values for finer-grained search
- Reduce parameter values for faster exploratory runs
- Focus on specific parameter ranges based on domain knowledge

PERFORMANCE:
Each backtest takes approximately 3-5 seconds on typical data.
- regime mode: ~48 configs × 4s = ~3 minutes
- phase mode: ~16 configs × 4s = ~1 minute  
- both mode: ~96 configs × 4s = ~6 minutes
Adjust parameter grids based on available time and compute resources.

WALK-FORWARD VALIDATION (BONUS):
If --walk-forward is specified with --train-bars and --test-bars:
- Split data into multiple train/test folds
- Train on train_bars, test on test_bars
- Report mean Sharpe and mean max drawdown across folds

Usage:
    python spot_bot/tune_mellin.py --csv data.csv --mode regime --top 5
    python spot_bot/tune_mellin.py --csv data.csv --mode phase --top 10 --out-csv results.csv
    python spot_bot/tune_mellin.py --csv data.csv --mode both --walk-forward --train-bars 1000 --test-bars 500
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path if running as script
if __package__ is None and __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spot_bot.backtest.backtest_spot import run_mean_reversion_backtests
from spot_bot.features import FeatureConfig


def load_ohlcv_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and validate OHLCV CSV with timestamp column.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with timestamp index and OHLCV columns
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)
    
    # Parse and set timestamp as index
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' column")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.set_index("timestamp")
    
    # Validate required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {missing}")
    
    return df[["open", "high", "low", "close", "volume"]]


def compute_additional_metrics(
    equity_curve: pd.Series,
    initial_equity: float,
    periods_per_year: float = 24 * 365,
) -> Dict[str, float]:
    """
    Compute additional performance metrics from equity curve.
    
    Args:
        equity_curve: Time series of equity values
        initial_equity: Starting equity value
        periods_per_year: Number of periods per year (default: hourly bars = 24*365)
        
    Returns:
        Dictionary with sharpe, volatility, and max_drawdown
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {
            "sharpe": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
        }
    
    # Compute returns
    returns = equity_curve.pct_change().dropna()
    
    # Sharpe ratio (annualized)
    if len(returns) > 0 and returns.std() != 0:
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0
    
    # Volatility (annualized)
    if len(returns) > 0:
        volatility = returns.std() * np.sqrt(periods_per_year)
    else:
        volatility = 0.0
    
    # Max drawdown
    peak = equity_curve.cummax()
    safe_peak = peak.where(peak > 0, 1e-8)
    drawdown = (equity_curve - peak) / safe_peak
    max_dd = float(drawdown.min())
    
    return {
        "sharpe": float(sharpe),
        "volatility": float(volatility),
        "max_drawdown": max_dd,
    }


def run_backtest_with_config(
    ohlcv_df: pd.DataFrame,
    config_dict: Dict[str, Any],
    slippage_bps: float,
    fee_rate: float,
    max_exposure: float,
    initial_equity: float,
) -> Dict[str, float]:
    """
    Run backtest with a specific configuration and return metrics.
    
    Args:
        ohlcv_df: OHLCV DataFrame
        config_dict: Dictionary of FeatureConfig parameters
        slippage_bps: Slippage in basis points
        fee_rate: Transaction fee rate
        max_exposure: Maximum exposure fraction
        initial_equity: Starting equity
        
    Returns:
        Dictionary of metrics including config parameters
    """
    # Create FeatureConfig from config_dict
    feature_config = FeatureConfig(**config_dict)
    
    # Run backtest
    try:
        results = run_mean_reversion_backtests(
            ohlcv_df=ohlcv_df,
            fee_rate=fee_rate,
            max_exposure=max_exposure,
            initial_equity=initial_equity,
            feature_config=feature_config,
            slippage_bps=slippage_bps,
        )
        
        # Use 'gated' (regime) results
        gated_result = results.get("gated")
        if gated_result is None:
            return None
        
        # Extract basic metrics
        metrics = dict(gated_result.metrics)
        
        # Compute additional metrics
        additional = compute_additional_metrics(
            gated_result.equity_curve,
            initial_equity,
        )
        metrics.update(additional)
        
        # Add configuration parameters to metrics
        metrics.update(config_dict)
        
        return metrics
        
    except Exception as e:
        print(f"Error running backtest with config {config_dict}: {e}", file=sys.stderr)
        return None


def generate_regime_grid() -> List[Dict[str, Any]]:
    """
    Generate parameter grid for regime mode (mellin_cepstrum).
    
    Returns:
        List of configuration dictionaries
    """
    # Fixed parameters
    base_config = {
        "psi_mode": "mellin_cepstrum",
        "psi_phase_agg": "peak",
    }
    
    # Tunable parameters - focused ranges for practical tuning
    # Users can modify these ranges based on their needs
    param_grid = {
        "psi_window": [128, 256, 512],
        "mellin_sigma": [0.0, 0.5],  # Reduced from 3 to 2 values
        "mellin_grid_n": [128, 256],  # Reduced from 3 to 2 values
        "psi_min_bin": [2, 4],  # Reduced from 3 to 2 values
        "psi_max_frac": [0.2, 0.25],  # Reduced from 4 to 2 values
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    configs = []
    for combo in itertools.product(*values):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        configs.append(config)
    
    return configs


def generate_phase_grid(base_configs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Generate parameter grid for phase mode (mellin_complex_cepstrum).
    
    Args:
        base_configs: Optional list of base configurations (e.g., from regime mode)
                     If None, uses a single default configuration
    
    Returns:
        List of configuration dictionaries
    """
    if base_configs is None:
        # Default base configuration
        base_configs = [{
            "psi_mode": "mellin_complex_cepstrum",
            "psi_window": 256,
            "mellin_grid_n": 256,
            "mellin_sigma": 0.0,
            "psi_min_bin": 2,
            "psi_max_frac": 0.25,
        }]
    
    # Tunable parameters for phase mode - focused ranges
    param_grid = {
        "mellin_detrend_phase": [True, False],
        "psi_phase_agg": ["peak", "cmean"],
        "psi_phase_power": [1.0, 1.5],  # Reduced from 4 to 2 values
        "mellin_eps": [1e-12, 1e-10],  # Reduced from 3 to 2 values
    }
    
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    configs = []
    for base_config in base_configs:
        for combo in itertools.product(*values):
            config = base_config.copy()
            config["psi_mode"] = "mellin_complex_cepstrum"
            config.update(dict(zip(keys, combo)))
            configs.append(config)
    
    return configs


def run_walk_forward(
    ohlcv_df: pd.DataFrame,
    config_dict: Dict[str, Any],
    train_bars: int,
    test_bars: int,
    slippage_bps: float,
    fee_rate: float,
    max_exposure: float,
    initial_equity: float,
) -> Dict[str, float]:
    """
    Run walk-forward validation for a configuration.
    
    Args:
        ohlcv_df: OHLCV DataFrame
        config_dict: Configuration dictionary
        train_bars: Number of bars for training
        test_bars: Number of bars for testing
        slippage_bps: Slippage in basis points
        fee_rate: Transaction fee rate
        max_exposure: Maximum exposure
        initial_equity: Starting equity
        
    Returns:
        Dictionary with mean metrics across folds
    """
    total_bars = len(ohlcv_df)
    fold_size = train_bars + test_bars
    
    if total_bars < fold_size:
        print(f"Warning: Not enough data for walk-forward. Need {fold_size}, have {total_bars}", file=sys.stderr)
        return None
    
    sharpe_scores = []
    max_dd_scores = []
    
    # Generate folds
    start = 0
    while start + fold_size <= total_bars:
        # Split into train and test
        test_start = start + train_bars
        test_end = test_start + test_bars
        
        # We only evaluate on test set
        test_df = ohlcv_df.iloc[test_start:test_end]
        
        # Run backtest on test set
        metrics = run_backtest_with_config(
            ohlcv_df=test_df,
            config_dict=config_dict,
            slippage_bps=slippage_bps,
            fee_rate=fee_rate,
            max_exposure=max_exposure,
            initial_equity=initial_equity,
        )
        
        if metrics is not None:
            sharpe_scores.append(metrics.get("sharpe", 0.0))
            max_dd_scores.append(metrics.get("max_drawdown", 0.0))
        
        # Move to next fold
        start += test_bars
    
    if not sharpe_scores:
        return None
    
    # Compute mean metrics
    result = config_dict.copy()
    result["mean_sharpe"] = float(np.mean(sharpe_scores))
    result["mean_max_drawdown"] = float(np.mean(max_dd_scores))
    result["num_folds"] = len(sharpe_scores)
    
    return result


def rank_configs(
    results_df: pd.DataFrame,
    rank_by: List[Tuple[str, bool]] = None,
) -> pd.DataFrame:
    """
    Rank configurations by performance metrics.
    
    Args:
        results_df: DataFrame with backtest results
        rank_by: List of (column, ascending) tuples for sorting
                 Default: [("sharpe", False), ("max_drawdown", False), ("turnover", True)]
    
    Returns:
        Sorted DataFrame
    """
    if rank_by is None:
        rank_by = [
            ("sharpe", False),  # Higher is better
            ("max_drawdown", False),  # Less negative is better
            ("turnover", True),  # Lower is better
        ]
    
    # Filter to only valid results
    valid_df = results_df.dropna(subset=[col for col, _ in rank_by])
    
    if valid_df.empty:
        return results_df
    
    # Sort by metrics
    sort_cols = [col for col, _ in rank_by]
    sort_ascending = [asc for _, asc in rank_by]
    
    ranked = valid_df.sort_values(by=sort_cols, ascending=sort_ascending)
    
    return ranked


def print_top_configs(results_df: pd.DataFrame, top_n: int = 5, walk_forward: bool = False) -> None:
    """
    Print top N configurations with their metrics.
    
    Args:
        results_df: Ranked DataFrame with results
        top_n: Number of top configurations to print
        walk_forward: Whether this is walk-forward validation
    """
    print(f"\n{'='*80}")
    print(f"TOP {top_n} CONFIGURATIONS")
    print(f"{'='*80}\n")
    
    # Key columns to display
    config_cols = [
        "psi_mode", "psi_window", "mellin_grid_n", "mellin_sigma",
        "psi_min_bin", "psi_max_frac", "psi_phase_agg",
        "mellin_detrend_phase", "psi_phase_power", "mellin_eps",
    ]
    
    if walk_forward:
        metric_cols = [
            "mean_sharpe", "mean_max_drawdown", "num_folds",
        ]
    else:
        metric_cols = [
            "sharpe", "final_return", "max_drawdown", "volatility", "turnover", "trades",
        ]
    
    for idx, row in results_df.head(top_n).iterrows():
        print(f"Rank {idx + 1}:")
        print(f"  Metrics:")
        for col in metric_cols:
            if col in row and pd.notna(row[col]):
                if col == "num_folds":
                    print(f"    {col:20s}: {int(row[col]):>10d}")
                else:
                    print(f"    {col:20s}: {row[col]:>10.4f}")
        
        print(f"  Configuration:")
        for col in config_cols:
            if col in row and pd.notna(row[col]):
                val = row[col]
                if isinstance(val, float):
                    print(f"    {col:20s}: {val:>10.4g}")
                else:
                    print(f"    {col:20s}: {str(val):>10s}")
        print()


def main() -> None:
    """Main entry point for parameter tuning script."""
    parser = argparse.ArgumentParser(
        description="Tune Mellin cepstrum parameters for regime detection and phase timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file")
    
    # Backtest parameters
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage in basis points")
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="Transaction fee rate")
    parser.add_argument("--max-exposure", type=float, default=1.0, help="Maximum exposure fraction")
    parser.add_argument("--initial-equity", type=float, default=1000.0, help="Initial equity")
    
    # Output parameters
    parser.add_argument("--out-csv", help="Path to save results CSV")
    parser.add_argument("--top", type=int, default=5, help="Number of top configurations to print")
    
    # Tuning mode
    parser.add_argument(
        "--mode",
        choices=["regime", "phase", "both"],
        default="regime",
        help="Tuning mode: regime (mellin_cepstrum), phase (mellin_complex_cepstrum), or both",
    )
    
    # Walk-forward validation
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward validation")
    parser.add_argument("--train-bars", type=int, default=1000, help="Training bars for walk-forward")
    parser.add_argument("--test-bars", type=int, default=500, help="Test bars for walk-forward")
    
    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading OHLCV data from {args.csv}...")
    try:
        ohlcv_df = load_ohlcv_csv(args.csv)
        print(f"Loaded {len(ohlcv_df)} bars from {ohlcv_df.index[0]} to {ohlcv_df.index[-1]}")
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate parameter grids based on mode
    all_configs = []
    
    if args.mode == "regime":
        print("\nGenerating parameter grid for regime mode (mellin_cepstrum)...")
        all_configs = generate_regime_grid()
        
    elif args.mode == "phase":
        print("\nGenerating parameter grid for phase mode (mellin_complex_cepstrum)...")
        all_configs = generate_phase_grid()
        
    elif args.mode == "both":
        print("\nGenerating parameter grids for both regime and phase modes...")
        regime_configs = generate_regime_grid()
        print(f"  Regime configs: {len(regime_configs)}")
        
        # Run regime sweep first (limit to first 10 for speed)
        print("  Running initial regime sweep to find best base configs...")
        regime_results = []
        num_initial = min(10, len(regime_configs))
        for i, config in enumerate(regime_configs[:num_initial]):
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i + 1}/{num_initial}", end="\r")
            
            metrics = run_backtest_with_config(
                ohlcv_df, config, args.slippage_bps, args.fee_rate,
                args.max_exposure, args.initial_equity,
            )
            if metrics is not None:
                regime_results.append(metrics)
        
        if regime_results:
            regime_df = pd.DataFrame(regime_results)
            ranked_regime = rank_configs(regime_df)
            
            # Use top 3 regime configs as base for phase sweep
            top_regime = ranked_regime.head(3)
            base_configs = []
            for _, row in top_regime.iterrows():
                base_config = {k: v for k, v in row.items() if k in [
                    "psi_window", "mellin_grid_n", "mellin_sigma",
                    "psi_min_bin", "psi_max_frac",
                ]}
                base_configs.append(base_config)
            
            phase_configs = generate_phase_grid(base_configs)
            print(f"  Phase configs (from {len(base_configs)} base configs): {len(phase_configs)}")
            
            # Combine all configs
            all_configs = regime_configs + phase_configs
        else:
            print("  Warning: No valid regime results, using default base for phase")
            all_configs = regime_configs + generate_phase_grid()
    
    print(f"\nTotal configurations to evaluate: {len(all_configs)}")
    
    # Run backtests for all configs
    results = []
    
    if args.walk_forward:
        print(f"\nRunning walk-forward validation (train={args.train_bars}, test={args.test_bars})...")
        rank_by = [("mean_sharpe", False), ("mean_max_drawdown", False)]
    else:
        print("\nRunning backtests...")
        rank_by = None  # Use default
    
    for i, config in enumerate(all_configs):
        if (i + 1) % 10 == 0 or (i + 1) == len(all_configs):
            print(f"Progress: {i + 1}/{len(all_configs)}", end="\r")
        
        if args.walk_forward:
            metrics = run_walk_forward(
                ohlcv_df, config, args.train_bars, args.test_bars,
                args.slippage_bps, args.fee_rate, args.max_exposure, args.initial_equity,
            )
        else:
            metrics = run_backtest_with_config(
                ohlcv_df, config, args.slippage_bps, args.fee_rate,
                args.max_exposure, args.initial_equity,
            )
        
        if metrics is not None:
            results.append(metrics)
    
    print()  # New line after progress
    
    if not results:
        print("No valid results generated!", file=sys.stderr)
        sys.exit(1)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Rank configurations
    print("\nRanking configurations...")
    ranked_df = rank_configs(results_df, rank_by)
    ranked_df = ranked_df.reset_index(drop=True)
    
    # Save to CSV if requested
    if args.out_csv:
        print(f"Saving results to {args.out_csv}...")
        ranked_df.to_csv(args.out_csv, index=False)
    
    # Print top configurations
    print_top_configs(ranked_df, args.top, args.walk_forward)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total configurations evaluated: {len(results)}")
    print(f"Valid results: {len(ranked_df)}")
    
    if not ranked_df.empty:
        if args.walk_forward:
            if "mean_sharpe" in ranked_df.columns:
                print(f"Mean Sharpe ratio range: [{ranked_df['mean_sharpe'].min():.4f}, {ranked_df['mean_sharpe'].max():.4f}]")
            if "mean_max_drawdown" in ranked_df.columns:
                print(f"Mean max drawdown range: [{ranked_df['mean_max_drawdown'].min():.4f}, {ranked_df['mean_max_drawdown'].max():.4f}]")
        else:
            if "sharpe" in ranked_df.columns:
                print(f"Sharpe ratio range: [{ranked_df['sharpe'].min():.4f}, {ranked_df['sharpe'].max():.4f}]")
            if "max_drawdown" in ranked_df.columns:
                print(f"Max drawdown range: [{ranked_df['max_drawdown'].min():.4f}, {ranked_df['max_drawdown'].max():.4f}]")
            if "final_return" in ranked_df.columns:
                print(f"Final return range: [{ranked_df['final_return'].min():.4f}, {ranked_df['final_return'].max():.4f}]")
    
    print("\nTuning complete!")


if __name__ == "__main__":
    main()
