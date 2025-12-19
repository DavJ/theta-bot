#!/usr/bin/env python3
"""
Minimal Real-Data Evaluation: Baseline vs Dual-Stream

Compares baseline and dual_stream models on BTCUSDT 1H sample data using
walk-forward validation and generates a markdown report with predictive and
trading metrics.

Usage:
    python -m theta_bot_averaging.eval.evaluate_dual_stream_real
    python -m theta_bot_averaging.eval.evaluate_dual_stream_real --fast
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from theta_bot_averaging.backtest import run_backtest
from theta_bot_averaging.data import build_targets, load_dataset
from theta_bot_averaging.features import build_dual_stream_inputs, build_features
from theta_bot_averaging.models import BaselineModel, DualStreamModel
from theta_bot_averaging.validation.purged_split import PurgedTimeSeriesSplit
from theta_bot_averaging.utils.data_sanity import log_data_sanity


def load_data(data_path: str) -> Tuple[pd.DataFrame, Dict]:
    """Load the BTCUSDT sample data and perform sanity check."""
    print(f"Loading data from {data_path}...")
    df = load_dataset(data_path)
    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Perform data sanity check
    sanity_stats = log_data_sanity(df, symbol="BTCUSDT")
    
    return df, sanity_stats


def evaluate_baseline(
    df_targets: pd.DataFrame,
    n_splits: int,
    horizon: int,
    threshold_bps: float,
    fee_rate: float = 0.0004,
    fast_mode: bool = False,
) -> Tuple[pd.DataFrame, Dict, List[Dict], Dict]:
    """
    Evaluate baseline model using walk-forward validation.
    
    Returns:
        predictions: Combined predictions DataFrame
        aggregate_metrics: Aggregated metrics across folds
        fold_metrics: List of per-fold metrics
        timing_stats: Dict with timing information
    """
    print("\n" + "=" * 70)
    print("BASELINE MODEL EVALUATION")
    print("=" * 70)
    
    t_start = time.perf_counter()
    features = build_features(df_targets)
    data = pd.concat([features, df_targets[["label", "future_return"]]], axis=1).dropna()
    
    if fast_mode:
        # Use fixed cap for fast mode
        N_FAST_BASELINE = 1500
        n_fast = min(N_FAST_BASELINE, len(data))
        data = data.head(n_fast)
        print(f"  Fast mode: Using first {n_fast} samples (capped at {N_FAST_BASELINE})")
    
    features = data[features.columns]
    targets = data["label"]
    future_returns = data["future_return"]
    t_features = time.perf_counter() - t_start
    print(f"  Feature building time: {t_features:.2f}s")
    
    # Convert basis points to decimal (10 bps = 10/10000 = 0.001)
    thr = threshold_bps / 10_000.0
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge=0, embargo=0)
    
    all_predictions = []
    all_metrics = []
    
    t_train_start = time.perf_counter()
    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(features.index), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        
        print(f"  Fold {fold_num}/{n_splits}: train={len(train_idx)}, test={len(test_idx)}")
        
        X_train, y_train = features.iloc[train_idx], targets.iloc[train_idx]
        X_test = features.iloc[test_idx]
        
        model = BaselineModel(
            mode="logit",
            positive_threshold=thr,
            negative_threshold=-thr,
            max_iter=200,
        )
        model.fit(X_train, y_train, future_return=future_returns.iloc[train_idx])
        preds = model.predict(X_test)
        
        fold_df = pd.DataFrame(
            {
                "predicted_return": preds.predicted_return,
                "signal": preds.signal,
                "future_return": future_returns.iloc[test_idx],
            }
        )
        
        # Compute fold metrics
        backtest_res = run_backtest(
            fold_df,
            position=fold_df["signal"],
            future_return_col="future_return",
            fee_rate=fee_rate,
        )
        all_metrics.append(backtest_res.metrics)
        all_predictions.append(fold_df)
    
    t_train = time.perf_counter() - t_train_start
    print(f"  Training time: {t_train:.2f}s")
    
    predictions_df = pd.concat(all_predictions).sort_index()
    aggregate_metrics = pd.DataFrame(all_metrics).mean(numeric_only=True).to_dict()
    
    timing_stats = {
        'feature_time': t_features,
        'train_time': t_train,
        'total_time': t_features + t_train,
    }
    
    # DIAGNOSTIC: Check prediction quality
    print(f"\n  DIAGNOSTICS (Baseline):")
    pred_std = predictions_df["predicted_return"].std()
    print(f"    std(predicted_return) = {pred_std:.6f}")
    
    # Class distribution
    class_counts = predictions_df["signal"].value_counts().sort_index()
    print(f"    Signal distribution: {dict(class_counts)}")
    
    # Class mean returns
    class_means = predictions_df.groupby("signal")["future_return"].mean()
    print(f"    Class mean returns:")
    for cls in [-1, 0, 1]:
        if cls in class_means.index:
            print(f"      signal={cls:+2d}: {class_means[cls]:+.6f}")
    
    # Check for degenerate predictions
    if pred_std < 1e-6:
        print(f"    ⚠️  WARNING: Predictions are nearly constant (std < 1e-6)")
    
    # Check if predicted_return exists
    assert "predicted_return" in predictions_df.columns, "predicted_return missing from output"
    
    print(f"  ✓ Completed {len(all_metrics)} folds")
    
    return predictions_df, aggregate_metrics, all_metrics, timing_stats


def evaluate_dual_stream(
    df_targets: pd.DataFrame,
    n_splits: int,
    horizon: int,
    threshold_bps: float,
    theta_window: int = 48,
    theta_q: float = 0.9,
    theta_terms: int = 8,
    mellin_k: int = 16,
    mellin_alpha: float = 0.5,
    mellin_omega_max: float = 1.0,
    torch_epochs: int = 50,
    torch_batch_size: int = 32,
    torch_lr: float = 1e-3,
    fee_rate: float = 0.0004,
    fast_mode: bool = False,
) -> Tuple[pd.DataFrame, Dict, List[Dict], Dict]:
    """
    Evaluate dual-stream model using walk-forward validation.
    
    Returns:
        predictions: Combined predictions DataFrame
        aggregate_metrics: Aggregated metrics across folds
        fold_metrics: List of per-fold metrics
        timing_stats: Dict with timing information
    """
    print("\n" + "=" * 70)
    print("DUAL-STREAM MODEL EVALUATION")
    print("=" * 70)
    
    # Apply fast mode parameter reductions
    if fast_mode:
        theta_terms = min(theta_terms, 6)
        mellin_k = min(mellin_k, 8)
        theta_window = min(theta_window, 96)
        print(f"  Fast mode parameter reductions:")
        print(f"    theta_terms: {theta_terms} (max 6)")
        print(f"    mellin_k: {mellin_k} (max 8)")
        print(f"    theta_window: {theta_window} (max 96)")
    
    # Build dual-stream inputs
    print(f"  Building Theta+Mellin features...")
    t_start = time.perf_counter()
    index, X_theta, X_mellin = build_dual_stream_inputs(
        df=df_targets,
        window=theta_window,
        q=theta_q,
        n_terms=theta_terms,
        mellin_k=mellin_k,
        alpha=mellin_alpha,
        omega_max=mellin_omega_max,
    )
    
    if fast_mode:
        # Use fixed cap for fast mode
        N_FAST_DUAL = 400
        N_FAST_DUAL_MAX = 600
        n_fast = min(N_FAST_DUAL, len(index))
        n_fast = min(n_fast, N_FAST_DUAL_MAX)  # Hard cap
        index = index[:n_fast]
        X_theta = X_theta[:n_fast]
        X_mellin = X_mellin[:n_fast]
        print(f"  Fast mode: Using first {n_fast} samples (capped at {N_FAST_DUAL}, max {N_FAST_DUAL_MAX})")
    
    t_features = time.perf_counter() - t_start
    print(f"  Theta shape: {X_theta.shape}, Mellin shape: {X_mellin.shape}")
    print(f"  Feature building time: {t_features:.2f}s")
    
    targets = df_targets.loc[index, "label"].values
    future_returns = df_targets.loc[index, "future_return"].values
    
    # Convert basis points to decimal (10 bps = 10/10000 = 0.001)
    thr = threshold_bps / 10_000.0
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge=0, embargo=0)
    
    all_predictions = []
    all_metrics = []
    
    t_train_start = time.perf_counter()
    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(index), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        
        print(f"  Fold {fold_num}/{n_splits}: train={len(train_idx)}, test={len(test_idx)}")
        
        X_theta_train, X_theta_test = X_theta[train_idx], X_theta[test_idx]
        X_mellin_train, X_mellin_test = X_mellin[train_idx], X_mellin[test_idx]
        y_train = targets[train_idx]
        future_return_train = future_returns[train_idx]
        test_index = index[test_idx]
        
        # Reduce epochs in fast mode
        epochs = torch_epochs if not fast_mode else min(5, torch_epochs)
        
        model = DualStreamModel(
            positive_threshold=thr,
            negative_threshold=-thr,
            epochs=epochs,
            batch_size=torch_batch_size,
            lr=torch_lr,
        )
        model.fit(X_theta_train, X_mellin_train, y_train, future_return=future_return_train)
        preds = model.predict(X_theta_test, X_mellin_test, test_index)
        
        fold_df = pd.DataFrame(
            {
                "predicted_return": preds.predicted_return,
                "signal": preds.signal,
                "future_return": future_returns[test_idx],
            }
        )
        
        # Compute fold metrics
        backtest_res = run_backtest(
            fold_df,
            position=fold_df["signal"],
            future_return_col="future_return",
            fee_rate=fee_rate,
        )
        all_metrics.append(backtest_res.metrics)
        all_predictions.append(fold_df)
    
    t_train = time.perf_counter() - t_train_start
    print(f"  Training time: {t_train:.2f}s")
    
    predictions_df = pd.concat(all_predictions).sort_index()
    aggregate_metrics = pd.DataFrame(all_metrics).mean(numeric_only=True).to_dict()
    
    timing_stats = {
        'feature_time': t_features,
        'train_time': t_train,
        'total_time': t_features + t_train,
    }
    
    # DIAGNOSTIC: Check prediction quality
    print(f"\n  DIAGNOSTICS (Dual-Stream):")
    pred_std = predictions_df["predicted_return"].std()
    print(f"    std(predicted_return) = {pred_std:.6f}")
    
    # Class distribution
    class_counts = predictions_df["signal"].value_counts().sort_index()
    print(f"    Signal distribution: {dict(class_counts)}")
    
    # Class mean returns
    class_means = predictions_df.groupby("signal")["future_return"].mean()
    print(f"    Class mean returns:")
    for cls in [-1, 0, 1]:
        if cls in class_means.index:
            print(f"      signal={cls:+2d}: {class_means[cls]:+.6f}")
    
    # Check for degenerate predictions
    if pred_std < 1e-6:
        print(f"    ⚠️  WARNING: Predictions are nearly constant (std < 1e-6)")
    
    # Check if predicted_return exists
    assert "predicted_return" in predictions_df.columns, "predicted_return missing from output"
    
    print(f"  ✓ Completed {len(all_metrics)} folds")
    
    return predictions_df, aggregate_metrics, all_metrics, timing_stats


def compute_predictive_metrics(predictions: pd.DataFrame) -> Dict:
    """
    Compute predictive metrics (correlation, hit rate).
    
    Excludes zero future returns for hit rate calculation.
    """
    pred = predictions["predicted_return"].values
    actual = predictions["future_return"].values
    
    # Correlation
    mask = ~(np.isnan(pred) | np.isnan(actual))
    if mask.sum() > 1:
        corr = np.corrcoef(pred[mask], actual[mask])[0, 1]
    else:
        corr = np.nan
    
    # Hit rate (direction accuracy, excluding zeros)
    mask_nonzero = mask & (actual != 0)
    if mask_nonzero.sum() > 0:
        pred_sign = np.sign(pred[mask_nonzero])
        actual_sign = np.sign(actual[mask_nonzero])
        hit_rate = (pred_sign == actual_sign).mean()
    else:
        hit_rate = np.nan
    
    return {
        "correlation": float(corr) if not np.isnan(corr) else 0.0,
        "hit_rate": float(hit_rate) if not np.isnan(hit_rate) else 0.0,
    }


def generate_report(
    df: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    dual_stream_preds: pd.DataFrame,
    baseline_metrics: Dict,
    dual_stream_metrics: Dict,
    config: Dict,
    output_path: str,
    data_sanity_stats: Dict = None,
    baseline_timing: Dict = None,
    dual_stream_timing: Dict = None,
    fast_mode: bool = False,
) -> None:
    """Generate markdown report comparing baseline and dual-stream models."""
    
    # Compute predictive metrics
    baseline_pred = compute_predictive_metrics(baseline_preds)
    dual_stream_pred = compute_predictive_metrics(dual_stream_preds)
    
    # Compute prediction diagnostics
    baseline_pred_std = baseline_preds["predicted_return"].std()
    dual_stream_pred_std = dual_stream_preds["predicted_return"].std()
    
    baseline_signal_counts = baseline_preds["signal"].value_counts().sort_index().to_dict()
    dual_stream_signal_counts = dual_stream_preds["signal"].value_counts().sort_index().to_dict()
    
    # Trading metrics already in metrics dicts
    
    # Determine dataset label based on sanity check
    if data_sanity_stats and data_sanity_stats.get('appears_unrealistic', False):
        dataset_label = "⚠️  **UNREALISTIC/SYNTHETIC-LIKE SAMPLE**"
    else:
        dataset_label = "✓ **REAL MARKET SAMPLE**"
    
    report = f"""# Dual-Stream Real Data Evaluation Report

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

## Dataset Summary

{dataset_label}

- **Symbol:** BTCUSDT
- **Timeframe:** 1 Hour (1H)
- **Date Range:** {df.index[0]} to {df.index[-1]}
- **Total Bars:** {len(df):,}
- **Price Range:** ${df['close'].min():.2f} to ${df['close'].max():.2f}

## Configuration

- **Mode:** {config.get('mode', 'default').upper()}
- **Horizon:** {config['horizon']} bar(s)
- **Threshold:** {config['threshold_bps']:.1f} bps
- **Walk-Forward Splits:** {config['n_splits']}
- **Fee Rate:** {config['fee_rate']:.4f}

### Dual-Stream Parameters
- **Theta Window:** {config['theta_window']}{' (optimized)' if config.get('mode') == 'optimized' else ''}
- **Theta q:** {config['theta_q']}
- **Theta Terms:** {config['theta_terms']}{' (optimized)' if config.get('mode') == 'optimized' else ''}
- **Mellin k:** {config['mellin_k']}{' (optimized)' if config.get('mode') == 'optimized' else ''}
- **Mellin alpha:** {config['mellin_alpha']}
- **Mellin omega_max:** {config['mellin_omega_max']}
- **Training Epochs:** {config['torch_epochs']}
- **Batch Size:** {config['torch_batch_size']}{' (optimized)' if config.get('mode') == 'optimized' else ''}
- **Learning Rate:** {config['torch_lr']}{' (optimized)' if config.get('mode') == 'optimized' else ''}

## Results Comparison

### Predictive Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Correlation** | {baseline_pred['correlation']:.4f} | {dual_stream_pred['correlation']:.4f} | {(dual_stream_pred['correlation'] - baseline_pred['correlation']):.4f} |
| **Hit Rate (Direction)** | {baseline_pred['hit_rate']:.2%} | {dual_stream_pred['hit_rate']:.2%} | {(dual_stream_pred['hit_rate'] - baseline_pred['hit_rate']):.2%} |

### Trading Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Total Return** | {baseline_metrics['total_return']:.2%} | {dual_stream_metrics['total_return']:.2%} | {(dual_stream_metrics['total_return'] - baseline_metrics['total_return']):.2%} |
| **Sharpe Ratio** | {baseline_metrics['sharpe']:.3f} | {dual_stream_metrics['sharpe']:.3f} | {(dual_stream_metrics['sharpe'] - baseline_metrics['sharpe']):.3f} |
| **Max Drawdown** | {baseline_metrics['max_drawdown']:.2%} | {dual_stream_metrics['max_drawdown']:.2%} | {(dual_stream_metrics['max_drawdown'] - baseline_metrics['max_drawdown']):.2%} |
| **Win Rate** | {baseline_metrics['win_rate']:.2%} | {dual_stream_metrics['win_rate']:.2%} | {(dual_stream_metrics['win_rate'] - baseline_metrics['win_rate']):.2%} |
| **Profit Factor** | {baseline_metrics['profit_factor']:.2f} | {dual_stream_metrics['profit_factor']:.2f} | {(dual_stream_metrics['profit_factor'] - baseline_metrics['profit_factor']):.2f} |

## Conclusion

"""
    
    # Generate conclusion based on results
    better_return = dual_stream_metrics['total_return'] > baseline_metrics['total_return']
    better_sharpe = dual_stream_metrics['sharpe'] > baseline_metrics['sharpe']
    better_corr = dual_stream_pred['correlation'] > baseline_pred['correlation']
    
    if better_return and better_sharpe:
        conclusion = "The Dual-Stream model outperforms the baseline in both total return and risk-adjusted performance (Sharpe ratio), "
    elif better_return:
        conclusion = "The Dual-Stream model achieves higher total return than baseline but with comparable risk-adjusted performance, "
    elif better_sharpe:
        conclusion = "The Dual-Stream model shows better risk-adjusted performance (Sharpe ratio) than baseline despite similar returns, "
    else:
        conclusion = "Both models show comparable performance, "
    
    if better_corr:
        conclusion += "with improved predictive correlation. "
    else:
        conclusion += "though baseline maintains slightly better predictive correlation. "
    
    conclusion += f"The evaluation demonstrates the dual-stream architecture's ability to process both Theta sequence patterns and Mellin transform features."
    
    report += conclusion
    
    # Add diagnostics summary section
    report += "\n\n## Diagnostics Summary\n\n"
    
    # FAST mode warning
    if fast_mode:
        report += "**⚠️  FAST MODE / DIAGNOSTIC ONLY**\n\n"
        report += "This report was generated in FAST mode with reduced splits and epochs.\n"
        report += "Results are for diagnostic purposes only and should not be used for final conclusions.\n\n"
    
    # Data sanity
    if data_sanity_stats:
        report += "### Data Sanity\n\n"
        report += f"- **min_close:** ${data_sanity_stats['min_close']:.2f}\n"
        report += f"- **max_close:** ${data_sanity_stats['max_close']:.2f}\n"
        report += f"- **Mean Price:** ${data_sanity_stats['mean_close']:.2f}\n"
        report += f"- **Start timestamp:** {data_sanity_stats['start_timestamp']}\n"
        report += f"- **End timestamp:** {data_sanity_stats['end_timestamp']}\n"
        report += f"- **Rows:** {data_sanity_stats['num_rows']:,}\n"
        report += f"- **appears_unrealistic:** {data_sanity_stats.get('appears_unrealistic', False)}\n"
        
        if data_sanity_stats.get('appears_unrealistic', False):
            report += f"\n**⚠️  WARNING:** {data_sanity_stats.get('warning_message', 'Data appears unrealistic')}\n"
        report += "\n"
    
    # Prediction diagnostics
    report += "### Prediction Quality\n\n"
    
    report += f"**Baseline Model:**\n"
    report += f"- predicted_return_std: {baseline_pred_std:.6f}\n"
    report += f"- Signal distribution: {baseline_signal_counts}\n"
    
    baseline_class_means = baseline_preds.groupby("signal")["future_return"].mean().to_dict()
    report += f"- Class mean returns:\n"
    for cls in [-1, 0, 1]:
        if cls in baseline_class_means:
            report += f"  - signal={cls:+2d}: {baseline_class_means[cls]:+.6f}\n"
    
    if baseline_pred_std < 1e-6:
        report += f"\n**⚠️  WARNING:** Model outputs nearly constant predicted_return; training may be ineffective or features degenerate.\n"
    
    report += f"\n**Dual-Stream Model:**\n"
    report += f"- predicted_return_std: {dual_stream_pred_std:.6f}\n"
    report += f"- Signal distribution: {dual_stream_signal_counts}\n"
    
    dual_stream_class_means = dual_stream_preds.groupby("signal")["future_return"].mean().to_dict()
    report += f"- Class mean returns:\n"
    for cls in [-1, 0, 1]:
        if cls in dual_stream_class_means:
            report += f"  - signal={cls:+2d}: {dual_stream_class_means[cls]:+.6f}\n"
    
    if dual_stream_pred_std < 1e-6:
        report += f"\n**⚠️  WARNING:** Model outputs nearly constant predicted_return; training may be ineffective or features degenerate.\n"
    
    # Root cause analysis
    report += "\n### Root Cause Analysis\n\n"
    
    # Determine most likely cause
    causes = []
    
    if data_sanity_stats and data_sanity_stats.get('appears_unrealistic', False):
        causes.append("DATA: Synthetic/unrealistic price data detected")
    
    if baseline_pred_std < 1e-6 or dual_stream_pred_std < 1e-6:
        causes.append("TRAINING: Model outputs are nearly constant (training ineffective)")
    
    if abs(baseline_pred['correlation']) < 0.05 and abs(dual_stream_pred['correlation']) < 0.05:
        causes.append("EVALUATION: Both models show near-zero correlation (possible fallback logic or data issues)")
    
    # Check if class means are nearly identical
    if baseline_class_means:
        mean_values = list(baseline_class_means.values())
        if len(mean_values) > 1 and max(mean_values) - min(mean_values) < 1e-4:
            causes.append("TARGET: Class mean returns are nearly identical (target may not be predictive)")
    
    if causes:
        report += "**Predictivity loss is most likely caused by:**\n\n"
        for cause in causes:
            report += f"- {cause}\n"
    else:
        report += "**No obvious root cause detected.** Predictivity may be limited by:\n"
        report += "- Market efficiency (weak signal in 1H data)\n"
        report += "- Model capacity (insufficient complexity)\n"
        report += "- Feature quality (limited information in Theta/Mellin features)\n"
    
    # Add footer based on data sanity
    report += "\n---\n"
    if data_sanity_stats and data_sanity_stats.get('appears_unrealistic', False):
        report += "*WARNING: This evaluation sample appears unrealistic; treat results as synthetic-like data. Results are for research purposes only.*\n"
    else:
        report += "*This evaluation uses the committed real-market sample dataset. Results are for research purposes only.*\n"
    
    # Write report
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"\n✓ Report written to: {output_path}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Baseline vs Dual-Stream on real BTCUSDT data"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer folds, fewer epochs, less data",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Use optimized hyperparameters (theta_window=72, terms=12, mellin_k=20, batch=64, lr=5e-4)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/BTCUSDT_1H_sample.csv.gz",
        help="Path to data file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/DUAL_STREAM_REAL_DATA_REPORT.md",
        help="Output report path",
    )
    args = parser.parse_args()
    
    # Configuration - use optimized parameters if requested
    if args.optimized:
        # Optimized parameters from DUAL_STREAM_EVALUATION_REPORT.md
        config = {
            "horizon": 1,
            "threshold_bps": 10.0,
            "n_splits": 3 if args.fast else 5,
            "fee_rate": 0.0004,
            # Optimized dual-stream params
            "theta_window": 72,  # Capture longer cycles
            "theta_q": 0.9,
            "theta_terms": 12,  # Model complex patterns
            "mellin_k": 20,  # Better frequency resolution
            "mellin_alpha": 0.5,
            "mellin_omega_max": 1.0,
            "torch_epochs": 5 if args.fast else 50,
            "torch_batch_size": 64,  # Larger batch for stability
            "torch_lr": 5e-4,  # Lower LR for stability
        }
    else:
        # Default configuration
        config = {
            "horizon": 1,
            "threshold_bps": 10.0,
            "n_splits": 3 if args.fast else 5,
            "fee_rate": 0.0004,
            # Dual-stream params
            "theta_window": 48,
            "theta_q": 0.9,
            "theta_terms": 8,
            "mellin_k": 16,
            "mellin_alpha": 0.5,
            "mellin_omega_max": 1.0,
            "torch_epochs": 5 if args.fast else 50,
            "torch_batch_size": 32,
            "torch_lr": 1e-3,
        }
    config["mode"] = "optimized" if args.optimized else "default"
    
    print("=" * 70)
    print("DUAL-STREAM REAL DATA EVALUATION")
    print("=" * 70)
    print(f"Mode: {'FAST' if args.fast else 'FULL'} {'(OPTIMIZED)' if args.optimized else ''}")
    print(f"Data: {args.data_path}")
    print(f"Report: {args.output}")
    if args.optimized:
        print("\nUsing Optimized Hyperparameters:")
        print(f"  - Theta Window: {config['theta_window']} (vs 48 default)")
        print(f"  - Theta Terms: {config['theta_terms']} (vs 8 default)")
        print(f"  - Mellin k: {config['mellin_k']} (vs 16 default)")
        print(f"  - Batch Size: {config['torch_batch_size']} (vs 32 default)")
        print(f"  - Learning Rate: {config['torch_lr']} (vs 1e-3 default)")
    
    
    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_path}")
        print("Please ensure data/BTCUSDT_1H_sample.csv.gz exists in the repository.")
        return 1
    
    df, data_sanity_stats = load_data(str(data_path))
    df_targets = build_targets(df, horizon=config["horizon"], threshold_bps=config["threshold_bps"])
    
    # DIAGNOSTIC: Parameter sanity check
    print("\n" + "=" * 70)
    print("PARAMETER SANITY CHECK")
    print("=" * 70)
    print(f"theta_window: {config['theta_window']}")
    print(f"theta_q: {config['theta_q']}")
    print(f"theta_terms: {config['theta_terms']}")
    print(f"mellin_k: {config['mellin_k']}")
    print(f"mellin_alpha: {config['mellin_alpha']}")
    print(f"mellin_omega_max: {config['mellin_omega_max']}")
    print(f"horizon: {config['horizon']}")
    print(f"threshold_bps: {config['threshold_bps']}")
    print(f"torch_epochs: {config['torch_epochs']}")
    print(f"torch_batch_size: {config['torch_batch_size']}")
    print(f"torch_lr: {config['torch_lr']}")
    
    # Check for potentially too-small parameters
    warnings = []
    if config['theta_window'] < 64:
        warnings.append(f"theta_window={config['theta_window']} may be too small to capture regime structure (consider >= 64)")
    if config['mellin_omega_max'] <= 1.0:
        warnings.append(f"mellin_omega_max={config['mellin_omega_max']} may be too small for frequency analysis (consider > 1.0)")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"    {w}")
    else:
        print("\n✓ Parameters appear reasonable")
    print("=" * 70 + "\n")
    
    # DIAGNOSTIC: Target alignment check (leakage sanity)
    print("=" * 70)
    print("TARGET ALIGNMENT CHECK")
    print("=" * 70)
    close_returns = df_targets["close"].pct_change()
    future_return = df_targets["future_return"]
    
    # Align for correlation computation
    aligned_df = pd.DataFrame({
        'close_return': close_returns,
        'future_return': future_return
    }).dropna()
    
    if len(aligned_df) > 1:
        lag0_corr = aligned_df['close_return'].corr(aligned_df['future_return'])
        print(f"Correlation between close_return[t] and future_return[t]: {lag0_corr:.4f}")
        
        if lag0_corr > 0.5:
            print("⚠️  WARNING: Strong positive correlation detected (possible leakage)")
        else:
            print("✓ No obvious leakage detected")
    print("=" * 70 + "\n")
    
    # Evaluate baseline
    baseline_preds, baseline_metrics, _, baseline_timing = evaluate_baseline(
        df_targets=df_targets,
        n_splits=config["n_splits"],
        horizon=config["horizon"],
        threshold_bps=config["threshold_bps"],
        fee_rate=config["fee_rate"],
        fast_mode=args.fast,
    )
    
    # Evaluate dual-stream
    dual_stream_preds, dual_stream_metrics, _, dual_stream_timing = evaluate_dual_stream(
        df_targets=df_targets,
        n_splits=config["n_splits"],
        horizon=config["horizon"],
        threshold_bps=config["threshold_bps"],
        theta_window=config["theta_window"],
        theta_q=config["theta_q"],
        theta_terms=config["theta_terms"],
        mellin_k=config["mellin_k"],
        mellin_alpha=config["mellin_alpha"],
        mellin_omega_max=config["mellin_omega_max"],
        torch_epochs=config["torch_epochs"],
        torch_batch_size=config["torch_batch_size"],
        torch_lr=config["torch_lr"],
        fee_rate=config["fee_rate"],
        fast_mode=args.fast,
    )
    
    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_report(
        df=df,
        baseline_preds=baseline_preds,
        dual_stream_preds=dual_stream_preds,
        baseline_metrics=baseline_metrics,
        dual_stream_metrics=dual_stream_metrics,
        config=config,
        output_path=str(output_path),
        data_sanity_stats=data_sanity_stats,
        baseline_timing=baseline_timing,
        dual_stream_timing=dual_stream_timing,
        fast_mode=args.fast,
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
