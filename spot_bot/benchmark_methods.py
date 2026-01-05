#!/usr/bin/env python3
"""
Multi-Pair Multi-Method Benchmark Runner using scale-phase psi.

This script compares different FeatureConfig parameter combinations
across multiple trading pairs and outputs clear summary tables showing
which configurations perform best.

FUNCTIONALITY:
- Load OHLCV data from CSVs (one per trading pair)
- Compare scale-phase variants (baseline mode) or run parameter grids (grid mode)
- Evaluate each (pair, method) combination via backtest
- Compute comprehensive metrics: Sharpe ratio, volatility, max drawdown, turnover, etc.
- Rank methods using composite score
- Generate leaderboards aggregated by method and by pair
- Support walk-forward validation for robustness testing

METHODS:
Baseline mode compares a single psi_mode: scale_phase.

Grid mode runs a small parameter grid to keep runtime reasonable:
- psi_window: [128, 256]
- base: [10.0, 2.0]

COMPOSITE SCORING:
score = sharpe - 0.5*abs(max_drawdown) - 0.05*turnover_norm
where turnover_norm = turnover / median(turnover across all runs)

OUTPUTS:
- results_runs.csv: All individual (pair, method) results
- leaderboard_methods.csv: Aggregated performance by method across all pairs
- leaderboard_pairs.csv: Aggregated performance by pair across all methods
- Console output showing top N methods and runs

Usage Examples:
    # Baseline comparison across pairs
    python spot_bot/benchmark_methods.py \\
        --data-dir data/ohlcv \\
        --pairs BTCUSDT,ETHUSDT,BNBUSDT \\
        --timeframe 1h \\
        --fees 0.0005 \\
        --out-dir results/benchmark \\
        --top 10 \\
        --mode baseline

    # Grid search with specific date range
    python spot_bot/benchmark_methods.py \\
        --data-dir data/ohlcv \\
        --timeframe 4h \\
        --start 2023-01-01 \\
        --end 2023-12-31 \\
        --fees 0.001 \\
        --slippage-bps 2.0 \\
        --out-dir results/grid \\
        --mode grid

    # Walk-forward validation
    python spot_bot/benchmark_methods.py \\
        --data-dir data/ohlcv \\
        --pairs BTCUSDT,ETHUSDT \\
        --timeframe 1h \\
        --out-dir results/walkforward \\
        --mode baseline \\
        --walk-forward \\
        --train-bars 1000 \\
        --test-bars 500
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path if running as script
if __package__ is None and __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spot_bot.backtest.backtest_spot import run_mean_reversion_backtests
from spot_bot.features import FeatureConfig


# Annualization factors for different timeframes
TIMEFRAME_PERIODS_PER_YEAR = {
    "1h": 24 * 365,
    "15m": 4 * 24 * 365,
    "4h": 6 * 365,
}


def load_pair_csv(
    csv_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV CSV for a single trading pair.
    
    Args:
        csv_path: Path to CSV file
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with datetime index and OHLCV columns
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)
    
    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    else:
        # Try to parse index as timestamp
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    
    # Validate required columns
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} must contain columns: {missing}")
    
    # Apply date filters
    if start_date is not None:
        start_ts = pd.Timestamp(start_date, tz="UTC")
        df = df[df.index >= start_ts]
    
    if end_date is not None:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[df.index <= end_ts]
    
    return df[["open", "high", "low", "close", "volume"]]


def infer_pairs_from_dir(data_dir: Path, timeframe: str) -> List[str]:
    """
    Infer trading pairs from CSV filenames in directory.
    
    Expected filename format: {PAIR}_{TIMEFRAME}.csv
    e.g., BTCUSDT_1h.csv, ETHUSDT_4h.csv
    
    Args:
        data_dir: Directory containing CSV files
        timeframe: Timeframe to filter (e.g., '1h', '4h')
        
    Returns:
        List of pair names (e.g., ['BTCUSDT', 'ETHUSDT'])
    """
    pairs = []
    pattern = f"*_{timeframe}.csv"
    
    for csv_path in sorted(data_dir.glob(pattern)):
        # Extract pair name from filename
        filename = csv_path.stem  # e.g., 'BTCUSDT_1h'
        pair = filename.rsplit("_", 1)[0]  # e.g., 'BTCUSDT'
        pairs.append(pair)
    
    return pairs


def generate_baseline_configs() -> List[Dict[str, Any]]:
    """
    Generate baseline method configurations.
    
    Returns a single config for scale-phase psi at default parameters.
    
    Returns:
        List of configuration dictionaries
    """
    return [{"psi_mode": "scale_phase"}]


def generate_grid_configs() -> List[Dict[str, Any]]:
    """
    Generate parameter grid configurations.
    
    Runs a small grid over scale-phase parameters to keep runtime reasonable:
    - psi_window: [128, 256]
    - base: [10.0, 2.0]
    
    Returns:
        List of configuration dictionaries
    """
    configs = []
    param_grid = {
        "psi_window": [128, 256],
        "base": [10.0, 2.0],
    }

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    for combo in itertools.product(*values):
        config = {"psi_mode": "scale_phase"}
        config.update(dict(zip(keys, combo)))
        configs.append(config)

    return configs


def compute_metrics(
    equity_curve: pd.Series,
    initial_equity: float,
    periods_per_year: float,
    base_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute comprehensive performance metrics.
    
    Args:
        equity_curve: Time series of equity values
        initial_equity: Starting equity
        periods_per_year: Annualization factor
        base_metrics: Base metrics from backtest (turnover, trades, etc.)
        
    Returns:
        Dictionary with all metrics including sharpe, volatility, max_drawdown
    """
    metrics = dict(base_metrics)
    
    if equity_curve.empty or len(equity_curve) < 2:
        metrics.update({
            "sharpe": 0.0,
            "volatility": 0.0,
            "max_drawdown": 0.0,
        })
        return metrics
    
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
    
    metrics.update({
        "sharpe": float(sharpe),
        "volatility": float(volatility),
        "max_drawdown": max_dd,
    })
    
    return metrics


def run_single_backtest(
    pair: str,
    ohlcv_df: pd.DataFrame,
    config_dict: Dict[str, Any],
    fee_rate: float,
    slippage_bps: float,
    initial_equity: float,
    periods_per_year: float,
) -> Optional[Dict[str, Any]]:
    """
    Run backtest for a single (pair, config) combination.
    
    Args:
        pair: Trading pair name
        ohlcv_df: OHLCV DataFrame
        config_dict: Configuration dictionary
        fee_rate: Transaction fee rate
        slippage_bps: Slippage in basis points
        initial_equity: Starting equity
        periods_per_year: Annualization factor
        
    Returns:
        Dictionary with results or None on error
    """
    try:
        # Create FeatureConfig
        feature_config = FeatureConfig(**config_dict)
        
        # Run backtest
        results = run_mean_reversion_backtests(
            ohlcv_df=ohlcv_df,
            fee_rate=fee_rate,
            max_exposure=1.0,
            initial_equity=initial_equity,
            feature_config=feature_config,
            slippage_bps=slippage_bps,
        )
        
        # Use 'gated' results (with regime detection)
        gated_result = results.get("gated")
        if gated_result is None:
            return None
        
        # Compute metrics
        metrics = compute_metrics(
            equity_curve=gated_result.equity_curve,
            initial_equity=initial_equity,
            periods_per_year=periods_per_year,
            base_metrics=gated_result.metrics,
        )
        
        # Add pair and config info
        result = {
            "pair": pair,
            "method_name": f"{config_dict['psi_mode']}",
            "psi_mode": config_dict["psi_mode"],
            "params_json": json.dumps(config_dict, sort_keys=True),
        }
        result.update(metrics)
        result.update(config_dict)
        
        return result
        
    except Exception as e:
        # Return error result instead of crashing
        error_result = {
            "pair": pair,
            "method_name": f"{config_dict.get('psi_mode', 'unknown')}",
            "psi_mode": config_dict.get("psi_mode", "unknown"),
            "params_json": json.dumps(config_dict, sort_keys=True),
            "error": str(e),
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "final_return": 0.0,
            "volatility": 0.0,
            "turnover": 0.0,
            "trades": 0,
        }
        error_result.update(config_dict)
        return error_result


def run_walk_forward_single(
    pair: str,
    ohlcv_df: pd.DataFrame,
    config_dict: Dict[str, Any],
    train_bars: int,
    test_bars: int,
    fee_rate: float,
    slippage_bps: float,
    initial_equity: float,
    periods_per_year: float,
) -> Optional[Dict[str, Any]]:
    """
    Run walk-forward validation for a single configuration.
    
    Args:
        pair: Trading pair name
        ohlcv_df: OHLCV DataFrame
        config_dict: Configuration dictionary
        train_bars: Number of bars for training window
        test_bars: Number of bars for testing window
        fee_rate: Transaction fee rate
        slippage_bps: Slippage in basis points
        initial_equity: Starting equity
        periods_per_year: Annualization factor
        
    Returns:
        Dictionary with mean metrics across folds or None on error
    """
    total_bars = len(ohlcv_df)
    fold_size = train_bars + test_bars
    
    if total_bars < fold_size:
        return None
    
    sharpe_scores = []
    max_dd_scores = []
    returns_scores = []
    
    # Generate folds
    start = 0
    while start + fold_size <= total_bars:
        # Test set starts after training window
        test_start = start + train_bars
        test_end = test_start + test_bars
        
        # Run backtest on test set only
        test_df = ohlcv_df.iloc[test_start:test_end]
        
        result = run_single_backtest(
            pair=pair,
            ohlcv_df=test_df,
            config_dict=config_dict,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            initial_equity=initial_equity,
            periods_per_year=periods_per_year,
        )
        
        if result is not None and "error" not in result:
            sharpe_scores.append(result.get("sharpe", 0.0))
            max_dd_scores.append(result.get("max_drawdown", 0.0))
            returns_scores.append(result.get("final_return", 0.0))
        
        # Move to next fold
        start += test_bars
    
    if not sharpe_scores:
        return None
    
    # Return mean metrics across folds
    result = {
        "pair": pair,
        "method_name": f"{config_dict['psi_mode']}",
        "psi_mode": config_dict["psi_mode"],
        "params_json": json.dumps(config_dict, sort_keys=True),
        "mean_sharpe": float(np.mean(sharpe_scores)),
        "mean_max_drawdown": float(np.mean(max_dd_scores)),
        "mean_final_return": float(np.mean(returns_scores)),
        "num_folds": len(sharpe_scores),
    }
    result.update(config_dict)
    
    return result


def compute_composite_score(
    results_df: pd.DataFrame,
    walk_forward: bool = False,
) -> pd.Series:
    """
    Compute composite score for ranking methods.
    
    Score formula:
    score = sharpe - 0.5*abs(max_drawdown) - 0.05*turnover_norm
    
    For walk-forward mode:
    score = mean_sharpe - 0.5*abs(mean_max_drawdown)
    
    Args:
        results_df: DataFrame with results
        walk_forward: Whether this is walk-forward validation
        
    Returns:
        Series with composite scores
    """
    if walk_forward:
        # Walk-forward scoring
        sharpe = results_df["mean_sharpe"].fillna(0.0)
        max_dd = results_df["mean_max_drawdown"].fillna(0.0)
        score = sharpe - 0.5 * np.abs(max_dd)
    else:
        # Standard scoring
        sharpe = results_df["sharpe"].fillna(0.0)
        max_dd = results_df["max_drawdown"].fillna(0.0)
        turnover = results_df["turnover"].fillna(0.0)
        
        # Normalize turnover by median
        median_turnover = turnover.median()
        if median_turnover > 0:
            turnover_norm = turnover / median_turnover
        else:
            turnover_norm = turnover
        
        score = sharpe - 0.5 * np.abs(max_dd) - 0.05 * turnover_norm
    
    return score


def create_leaderboards(
    results_df: pd.DataFrame,
    walk_forward: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create aggregated leaderboards by method and by pair.
    
    Args:
        results_df: DataFrame with all results
        walk_forward: Whether this is walk-forward validation
        
    Returns:
        Tuple of (methods_leaderboard, pairs_leaderboard)
    """
    # Add composite score
    results_df["score"] = compute_composite_score(results_df, walk_forward)
    
    if walk_forward:
        agg_cols = {
            "score": ["mean", "median", "std"],
            "mean_sharpe": ["mean", "median"],
            "mean_max_drawdown": ["mean", "median"],
            "mean_final_return": ["mean", "median"],
        }
    else:
        agg_cols = {
            "score": ["mean", "median", "std"],
            "sharpe": ["mean", "median"],
            "max_drawdown": ["mean", "median"],
            "final_return": ["mean", "median"],
            "volatility": ["mean", "median"],
            "turnover": ["mean", "median"],
        }
    
    # Leaderboard by method (across all pairs)
    methods_leaderboard = (
        results_df.groupby("method_name")
        .agg(agg_cols)
        .reset_index()
    )
    # Flatten column names
    methods_leaderboard.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in methods_leaderboard.columns.values
    ]
    methods_leaderboard = methods_leaderboard.sort_values("score_mean", ascending=False)
    
    # Leaderboard by pair (across all methods)
    pairs_leaderboard = (
        results_df.groupby("pair")
        .agg(agg_cols)
        .reset_index()
    )
    # Flatten column names
    pairs_leaderboard.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in pairs_leaderboard.columns.values
    ]
    pairs_leaderboard = pairs_leaderboard.sort_values("score_mean", ascending=False)
    
    return methods_leaderboard, pairs_leaderboard


def print_top_results(
    results_df: pd.DataFrame,
    methods_leaderboard: pd.DataFrame,
    top_n: int = 10,
    walk_forward: bool = False,
) -> None:
    """
    Print top N methods and top N individual runs.
    
    Args:
        results_df: DataFrame with all results (with score column)
        methods_leaderboard: Aggregated leaderboard by method
        top_n: Number of results to print
        walk_forward: Whether this is walk-forward validation
    """
    print("\n" + "=" * 80)
    print(f"TOP {top_n} METHODS (by mean score across pairs)")
    print("=" * 80 + "\n")
    
    for rank, (idx, row) in enumerate(methods_leaderboard.head(top_n).iterrows(), start=1):
        print(f"Rank {rank}: {row['method_name']}")
        print(f"  Mean Score: {row['score_mean']:.4f} (±{row.get('score_std', 0.0):.4f})")
        if walk_forward:
            print(f"  Mean Sharpe: {row['mean_sharpe_mean']:.4f}")
            print(f"  Mean Max DD: {row['mean_max_drawdown_mean']:.4f}")
            print(f"  Mean Return: {row['mean_final_return_mean']:.4f}")
        else:
            print(f"  Mean Sharpe: {row['sharpe_mean']:.4f}")
            print(f"  Mean Max DD: {row['max_drawdown_mean']:.4f}")
            print(f"  Mean Return: {row['final_return_mean']:.4f}")
        print()
    
    print("\n" + "=" * 80)
    print(f"TOP {top_n} INDIVIDUAL RUNS (by score)")
    print("=" * 80 + "\n")
    
    top_runs = results_df.nlargest(top_n, "score")
    for rank, (idx, row) in enumerate(top_runs.iterrows(), start=1):
        print(f"Rank {rank}: {row['pair']} + {row['method_name']}")
        print(f"  Score: {row['score']:.4f}")
        if walk_forward:
            print(f"  Mean Sharpe: {row['mean_sharpe']:.4f}")
            print(f"  Mean Max DD: {row['mean_max_drawdown']:.4f}")
            print(f"  Num Folds: {int(row['num_folds'])}")
        else:
            print(f"  Sharpe: {row['sharpe']:.4f}")
            print(f"  Max DD: {row['max_drawdown']:.4f}")
            print(f"  Return: {row['final_return']:.4f}")
        print()


def main() -> None:
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark multiple bot methods across multiple trading pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Data parameters
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory with OHLCV CSVs (one per pair, e.g., BTCUSDT_1h.csv)",
    )
    parser.add_argument(
        "--pairs",
        help="Comma-separated list of pairs (e.g., BTCUSDT,ETHUSDT). If not provided, infer from filenames.",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["1h", "15m", "4h"],
        help="Timeframe for data and annualization (default: 1h)",
    )
    parser.add_argument(
        "--start",
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        help="End date filter (YYYY-MM-DD)",
    )
    
    # Backtest parameters
    parser.add_argument(
        "--fees",
        type=float,
        default=0.0005,
        help="Transaction fee rate (default: 0.0005)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Slippage in basis points (default: 0.0)",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=1000.0,
        help="Initial equity (default: 1000.0)",
    )
    
    # Output parameters
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for CSV results",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results to print (default: 10)",
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["baseline", "grid"],
        default="baseline",
        help="Comparison mode: baseline (single scale_phase config) or grid (parameter search)",
    )
    
    # Walk-forward validation
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Enable walk-forward validation",
    )
    parser.add_argument(
        "--train-bars",
        type=int,
        default=1000,
        help="Training bars for walk-forward (default: 1000)",
    )
    parser.add_argument(
        "--test-bars",
        type=int,
        default=500,
        help="Test bars for walk-forward (default: 500)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data_dir.exists():
        print(f"Error: Data directory {args.data_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine pairs
    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(",")]
    else:
        pairs = infer_pairs_from_dir(args.data_dir, args.timeframe)
        if not pairs:
            print(f"Error: No CSV files found matching pattern *_{args.timeframe}.csv in {args.data_dir}", file=sys.stderr)
            sys.exit(1)
    
    print(f"Trading pairs: {', '.join(pairs)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Mode: {args.mode}")
    print(f"Walk-forward: {args.walk_forward}")
    
    # Get annualization factor
    periods_per_year = TIMEFRAME_PERIODS_PER_YEAR.get(args.timeframe, 24 * 365)
    
    # Generate configurations
    if args.mode == "baseline":
        configs = generate_baseline_configs()
    else:  # grid
        configs = generate_grid_configs()
    
    print(f"\nConfigurations to test: {len(configs)}")
    print(f"Total runs: {len(pairs)} pairs × {len(configs)} configs = {len(pairs) * len(configs)}")
    
    # Load data for all pairs
    pair_data = {}
    for pair in pairs:
        csv_path = args.data_dir / f"{pair}_{args.timeframe}.csv"
        if not csv_path.exists():
            print(f"Warning: CSV not found for {pair}: {csv_path}", file=sys.stderr)
            continue
        
        try:
            df = load_pair_csv(csv_path, args.start, args.end)
            pair_data[pair] = df
            print(f"Loaded {pair}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        except Exception as e:
            print(f"Error loading {pair}: {e}", file=sys.stderr)
    
    if not pair_data:
        print("Error: No valid data loaded", file=sys.stderr)
        sys.exit(1)
    
    # Run backtests
    print(f"\nRunning backtests...")
    results = []
    total_runs = len(pair_data) * len(configs)
    run_num = 0
    
    for pair, ohlcv_df in sorted(pair_data.items()):
        for config in configs:
            run_num += 1
            if run_num % 10 == 0 or run_num == total_runs:
                print(f"Progress: {run_num}/{total_runs}", end="\r")
            
            if args.walk_forward:
                result = run_walk_forward_single(
                    pair=pair,
                    ohlcv_df=ohlcv_df,
                    config_dict=config,
                    train_bars=args.train_bars,
                    test_bars=args.test_bars,
                    fee_rate=args.fees,
                    slippage_bps=args.slippage_bps,
                    initial_equity=args.initial_equity,
                    periods_per_year=periods_per_year,
                )
            else:
                result = run_single_backtest(
                    pair=pair,
                    ohlcv_df=ohlcv_df,
                    config_dict=config,
                    fee_rate=args.fees,
                    slippage_bps=args.slippage_bps,
                    initial_equity=args.initial_equity,
                    periods_per_year=periods_per_year,
                )
            
            if result is not None:
                results.append(result)
    
    print()  # New line after progress
    
    if not results:
        print("Error: No valid results generated", file=sys.stderr)
        sys.exit(1)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw runs
    runs_path = args.out_dir / "results_runs.csv"
    results_df.to_csv(runs_path, index=False)
    print(f"\nSaved raw results to {runs_path}")
    
    # Create leaderboards
    methods_leaderboard, pairs_leaderboard = create_leaderboards(results_df, args.walk_forward)
    
    # Save leaderboards
    methods_path = args.out_dir / "leaderboard_methods.csv"
    methods_leaderboard.to_csv(methods_path, index=False)
    print(f"Saved methods leaderboard to {methods_path}")
    
    pairs_path = args.out_dir / "leaderboard_pairs.csv"
    pairs_leaderboard.to_csv(pairs_path, index=False)
    print(f"Saved pairs leaderboard to {pairs_path}")
    
    # Print top results
    print_top_results(results_df, methods_leaderboard, args.top, args.walk_forward)
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
