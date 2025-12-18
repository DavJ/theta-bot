#!/usr/bin/env python
"""
Evaluation script for dual-stream theta + Mellin model predictivity.

This script evaluates and compares the predictive performance of:
1. Dual-stream theta + Mellin model
2. Baseline logistic regression model

Uses synthetic data with realistic market characteristics.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from theta_bot_averaging.validation import run_walkforward


def generate_realistic_market_data(n_samples=500, seed=42):
    """
    Generate synthetic market data with realistic characteristics:
    - Multiple frequency components (trend + cycles)
    - Realistic volatility
    - Occasional regime changes
    
    Args:
        n_samples: Number of hourly candles
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)
    
    print(f"Generating {n_samples} samples of synthetic market data...")
    
    idx = pd.date_range("2024-01-01", periods=n_samples, freq="h")
    t = np.linspace(0, 20 * np.pi, n_samples)
    
    # Base price with trend
    trend = 100 + 0.5 * t
    
    # Multiple cyclical components (different market rhythms)
    cycle1 = 30 * np.sin(t)                    # Main cycle
    cycle2 = 15 * np.sin(2.3 * t + 0.5)       # Harmonic
    cycle3 = 8 * np.sin(5.7 * t + 1.2)        # Higher frequency
    
    # Regime changes (simulate bull/bear transitions)
    regime = np.ones(n_samples)
    regime[n_samples//3:2*n_samples//3] = 0.5  # Bear phase
    
    # Combine components
    prices = trend + regime * (cycle1 + cycle2 + cycle3)
    
    # Add realistic noise
    noise = np.random.randn(n_samples) * 2.0
    prices = prices + noise
    
    # Ensure positive prices
    prices = np.maximum(prices, 50)
    
    # Create OHLCV
    close = prices
    high = close + np.abs(np.random.randn(n_samples)) * 1.5
    low = close - np.abs(np.random.randn(n_samples)) * 1.5
    open_price = close + np.random.randn(n_samples) * 0.5
    volume = 1000 + 500 * np.abs(np.cos(t)) + np.random.rand(n_samples) * 200
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=idx)
    
    print(f"✓ Generated data from {df.index[0]} to {df.index[-1]}")
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"  Mean return: {df['close'].pct_change().mean():.6f}")
    print(f"  Volatility: {df['close'].pct_change().std():.6f}")
    
    return df


def compute_correlation_and_metrics(predictions_df):
    """
    Compute prediction performance metrics.
    
    Args:
        predictions_df: DataFrame with predicted_return and future_return columns
    
    Returns:
        Dictionary of metrics
    """
    pred = predictions_df['predicted_return'].values
    actual = predictions_df['future_return'].values
    signal = predictions_df['signal'].values
    
    # Correlation
    if len(pred) > 1 and np.std(pred) > 0 and np.std(actual) > 0:
        corr = np.corrcoef(pred, actual)[0, 1]
    else:
        corr = 0.0
    
    # Hit rate (directional accuracy, excluding neutrals)
    actual_direction = np.sign(actual)
    correct = (signal * actual_direction > 0).sum()
    incorrect = (signal * actual_direction < 0).sum()
    neutral = (signal == 0).sum()
    total_trades = len(signal) - neutral
    hit_rate = correct / total_trades if total_trades > 0 else 0.0
    
    # Win/loss stats
    wins = (signal * actual > 0).sum()
    losses = (signal * actual < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    
    # Profit factor (gross profit / gross loss)
    gross_profit = (signal * actual)[signal * actual > 0].sum()
    gross_loss = abs((signal * actual)[signal * actual < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    # Cumulative return
    cumulative_return = (signal * actual).sum()
    
    # Sharpe ratio (simple version)
    returns = signal * actual
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    
    return {
        'correlation': corr,
        'hit_rate': hit_rate,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe,
        'total_trades': total_trades,
        'wins': int(wins),
        'losses': int(losses),
        'neutral_signals': int(neutral),
    }


def run_evaluation(n_samples=500, n_splits=3, output_dir='evaluation_results'):
    """
    Run evaluation comparing dual-stream and baseline models.
    
    Args:
        n_samples: Number of samples
        n_splits: Number of walk-forward splits
        output_dir: Directory for results
    
    Returns:
        Dictionary with comparison results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    df = generate_realistic_market_data(n_samples)
    
    # Save data
    data_file = output_path / "synthetic_market_data.csv"
    df.to_csv(data_file)
    print(f"✓ Data saved to {data_file}")
    
    results = {}
    
    # Evaluate both models
    for model_type in ['logit', 'dual_stream']:
        print(f"\n{'='*70}")
        print(f"Evaluating {model_type.upper()} model...")
        print('='*70)
        
        # Create config
        config = {
            'data_path': str(data_file),
            'horizon': 1,
            'threshold_bps': 10.0,
            'model_type': model_type,
            'fee_rate': 0.0004,
            'slippage_bps': 1.0,
            'spread_bps': 0.5,
            'n_splits': n_splits,
            'purge': 1,
            'embargo': 1,
            'output_dir': str(output_path / f"{model_type}_runs"),
        }
        
        # Add dual-stream specific params
        if model_type == 'dual_stream':
            config.update({
                'theta_window': 48,
                'theta_q': 0.9,
                'theta_terms': 8,
                'mellin_k': 16,
                'mellin_alpha': 0.5,
                'mellin_omega_max': 1.0,
                'torch_epochs': 30,
                'torch_batch_size': 32,
                'torch_lr': 1e-3,
            })
        
        # Save config
        config_file = output_path / f"config_{model_type}.yaml"
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f)
        
        # Run walk-forward validation
        try:
            result = run_walkforward(str(config_file))
            
            # Load predictions
            pred_files = list(Path(result['output_dir']).rglob('predictions.parquet'))
            if pred_files:
                preds = pd.read_parquet(pred_files[0])
                
                # Compute metrics
                metrics = compute_correlation_and_metrics(preds)
                
                # Add backtest metrics
                metrics.update({
                    'total_return': result['metrics'].get('total_return', 0),
                    'sharpe': result['metrics'].get('sharpe', 0),
                    'max_drawdown': result['metrics'].get('max_drawdown', 0),
                    'sortino': result['metrics'].get('sortino', 0),
                })
                
                results[model_type] = {
                    'metrics': metrics,
                    'predictions': preds,
                    'config': config,
                }
                
                print(f"\n{model_type.upper()} Results:")
                print(f"  Correlation:      {metrics['correlation']:.4f}")
                print(f"  Hit Rate:         {metrics['hit_rate']:.2%}")
                print(f"  Win Rate:         {metrics['win_rate']:.2%}")
                print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:.4f}")
                print(f"  Profit Factor:    {metrics['profit_factor']:.4f}")
                print(f"  Cumulative Ret:   {metrics['cumulative_return']:.4f}")
                print(f"  Total Trades:     {metrics['total_trades']}")
                print(f"  Wins/Losses:      {metrics['wins']}/{metrics['losses']}")
            else:
                print(f"✗ No predictions found for {model_type}")
                results[model_type] = None
                
        except Exception as e:
            print(f"✗ Error evaluating {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = None
    
    # Generate comparison report
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print('='*70)
    
    if results.get('logit') and results.get('dual_stream'):
        baseline = results['logit']['metrics']
        dual_stream = results['dual_stream']['metrics']
        
        print("\nMetric                  Baseline (logit)    Dual-Stream    Improvement")
        print("-" * 70)
        
        for metric in ['correlation', 'hit_rate', 'win_rate', 'sharpe_ratio', 
                       'profit_factor', 'cumulative_return']:
            base_val = baseline[metric]
            ds_val = dual_stream[metric]
            
            if base_val != 0:
                improvement = ((ds_val - base_val) / abs(base_val)) * 100
            else:
                improvement = 0 if ds_val == 0 else float('inf')
            
            print(f"{metric:23} {base_val:15.4f} {ds_val:15.4f} {improvement:+10.1f}%")
        
        # Save comparison to JSON (convert numpy types)
        comparison_file = output_path / 'comparison_results.json'
        
        def convert_numpy(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        comparison = {
            'data_type': 'synthetic',
            'n_samples': n_samples,
            'n_splits': n_splits,
            'baseline': convert_numpy(baseline),
            'dual_stream': convert_numpy(dual_stream),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n✓ Comparison saved to {comparison_file}")
        
        # Interpretation
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print('='*70)
        
        # Check if dual-stream shows improvement
        improvements = []
        for metric in ['correlation', 'hit_rate', 'sharpe_ratio']:
            base_val = baseline[metric]
            ds_val = dual_stream[metric]
            if ds_val > base_val:
                improvements.append(metric)
        
        if len(improvements) >= 2:
            print("\n✓ Dual-stream model shows POSITIVE predictivity improvement:")
            print(f"  Improved metrics: {', '.join(improvements)}")
        elif len(improvements) == 1:
            print("\n⚠ Dual-stream model shows MIXED results:")
            print(f"  Improved: {improvements[0]}")
            print("  Other metrics comparable or lower")
        else:
            print("\n✗ Dual-stream model shows NO significant improvement")
            print("  This could indicate:")
            print("  - Insufficient training data")
            print("  - Hyperparameter tuning needed")
            print("  - Model complexity not suited for this data")
        
        print("\nNote: Results on synthetic data may differ from real market data.")
        print("For production use, validate on real historical data from Binance.")
        
    else:
        print("\n✗ Could not generate comparison (missing results)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dual-stream model predictivity"
    )
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of samples (default: 500)')
    parser.add_argument('--n-splits', type=int, default=3,
                        help='Walk-forward splits (default: 3)')
    parser.add_argument('--output-dir', default='evaluation_results',
                        help='Output directory (default: evaluation_results)')
    
    args = parser.parse_args()
    
    print(f"""
{'='*70}
DUAL-STREAM THETA + MELLIN MODEL PREDICTIVITY EVALUATION
{'='*70}

Configuration:
  Data Type:     Synthetic (realistic market characteristics)
  Samples:       {args.n_samples}
  Splits:        {args.n_splits}
  Output:        {args.output_dir}

This script will:
1. Generate synthetic market data with multiple frequency components
2. Run walk-forward validation with baseline (logit) model
3. Run walk-forward validation with dual-stream model
4. Compare predictive performance metrics

{'='*70}
""")
    
    results = run_evaluation(
        n_samples=args.n_samples,
        n_splits=args.n_splits,
        output_dir=args.output_dir
    )
    
    if results:
        print(f"\n{'='*70}")
        print("✓ EVALUATION COMPLETE")
        print('='*70)
        print(f"\nResults saved to: {args.output_dir}/")
        print("  - comparison_results.json")
        print("  - config_logit.yaml")
        print("  - config_dual_stream.yaml")
        print("  - synthetic_market_data.csv")
        print("  - predictions and metrics in respective run directories")
    else:
        print("\n✗ Evaluation failed")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
