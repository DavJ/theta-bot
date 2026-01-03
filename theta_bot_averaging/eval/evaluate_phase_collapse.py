#!/usr/bin/env python3
"""
Phase-Collapse Event Prediction Evaluation

Evaluates VOL_BURST event prediction on real BTCUSDT data using baseline and
dual-stream models with walk-forward validation.

Usage:
    python -m theta_bot_averaging.eval.evaluate_phase_collapse
    python -m theta_bot_averaging.eval.evaluate_phase_collapse --fast
    python -m theta_bot_averaging.eval.evaluate_phase_collapse --event_type vol_burst --horizon 8 --quantile 0.80
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Small epsilon to avoid division by zero
EPSILON = 1e-9

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from theta_bot_averaging.data import load_dataset
from theta_bot_averaging.features import build_features, build_dual_stream_inputs
from theta_bot_averaging.targets import make_vol_burst_labels
from theta_bot_averaging.validation.purged_split import PurgedTimeSeriesSplit


def evaluate_baseline_events(
    df: pd.DataFrame,
    event_labels: pd.Series,
    future_vol: pd.Series,
    n_splits: int,
    quantile: float,
    fast_mode: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate baseline model on event prediction using walk-forward validation.
    
    Returns:
        aggregate_metrics: Aggregated metrics across folds
        fold_metrics: List of per-fold metrics with sanity checks
    """
    print("\n" + "=" * 70)
    print("BASELINE MODEL - EVENT PREDICTION")
    print("=" * 70)
    
    # Build features
    t_start = time.perf_counter()
    features = build_features(df)
    
    # Align features with event labels
    common_index = features.index.intersection(event_labels.index)
    features = features.loc[common_index]
    labels = event_labels.loc[common_index]
    future_vol_aligned = future_vol.loc[common_index]
    
    # Drop NaN features
    valid_mask = ~features.isna().any(axis=1)
    features = features[valid_mask]
    labels = labels[valid_mask]
    future_vol_aligned = future_vol_aligned[valid_mask]
    
    if fast_mode:
        # Cap samples for fast mode
        N_FAST = 1500
        n_fast = min(N_FAST, len(features))
        features = features.head(n_fast)
        labels = labels.head(n_fast)
        future_vol_aligned = future_vol_aligned.head(n_fast)
        print(f"  Fast mode: Using first {n_fast} samples (capped at {N_FAST})")
    
    t_features = time.perf_counter() - t_start
    print(f"  Feature building time: {t_features:.2f}s")
    print(f"  Samples: {len(features)}, Features: {features.shape[1]}")
    
    # Walk-forward validation
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge=0, embargo=0)
    
    all_metrics = []
    all_predictions = []
    
    t_train_start = time.perf_counter()
    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(features.index), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        
        print(f"  Fold {fold_num}/{n_splits}: train={len(train_idx)}, test={len(test_idx)}")
        
        X_train = features.iloc[train_idx].values
        y_train = labels.iloc[train_idx].values
        future_vol_train = future_vol_aligned.iloc[train_idx].values
        
        X_test = features.iloc[test_idx].values
        y_test = labels.iloc[test_idx].values
        future_vol_test = future_vol_aligned.iloc[test_idx].values
        
        # Compute fold-specific threshold on TRAIN data only
        train_threshold = np.quantile(future_vol_train, quantile)
        
        # Recompute labels with train threshold (fold-safe)
        y_train_foldsafe = (future_vol_train >= train_threshold).astype(int)
        y_test_foldsafe = (future_vol_test >= train_threshold).astype(int)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train LogisticRegression with balanced class weights
        model = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train_scaled, y_train_foldsafe)
        
        # Predict probabilities
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Compute metrics
        roc_auc = roc_auc_score(y_test_foldsafe, probs)
        pr_auc = average_precision_score(y_test_foldsafe, probs)
        brier = brier_score_loss(y_test_foldsafe, probs)
        event_rate = y_test_foldsafe.mean()
        
        # Gating sanity check: compare top 20% vs bottom 20%
        top_20_pct = np.quantile(probs, 0.80)
        bottom_20_pct = np.quantile(probs, 0.20)
        top_mask = probs >= top_20_pct
        bottom_mask = probs <= bottom_20_pct
        
        if top_mask.sum() > 0 and bottom_mask.sum() > 0:
            mean_vol_top = future_vol_test[top_mask].mean()
            mean_vol_bottom = future_vol_test[bottom_mask].mean()
            vol_ratio = mean_vol_top / (mean_vol_bottom + EPSILON)
            vol_diff = mean_vol_top - mean_vol_bottom
        else:
            mean_vol_top = np.nan
            mean_vol_bottom = np.nan
            vol_ratio = np.nan
            vol_diff = np.nan
        
        fold_metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "event_rate": event_rate,
            "mean_vol_top": mean_vol_top,
            "mean_vol_bottom": mean_vol_bottom,
            "vol_ratio": vol_ratio,
            "vol_diff": vol_diff,
        }
        all_metrics.append(fold_metrics)
        
        # Store predictions
        all_predictions.append({
            "probs": probs,
            "y_true": y_test_foldsafe,
            "future_vol": future_vol_test,
        })
    
    t_train = time.perf_counter() - t_train_start
    print(f"  Training time: {t_train:.2f}s")
    print(f"  ✓ Completed {len(all_metrics)} folds")
    
    # Aggregate metrics
    aggregate_metrics = pd.DataFrame(all_metrics).mean(numeric_only=True).to_dict()
    
    return aggregate_metrics, all_metrics


def evaluate_dual_stream_events(
    df: pd.DataFrame,
    event_labels: pd.Series,
    future_vol: pd.Series,
    n_splits: int,
    quantile: float,
    theta_window: int = 48,
    theta_q: float = 0.9,
    theta_terms: int = 8,
    mellin_k: int = 16,
    mellin_alpha: float = 0.5,
    mellin_omega_max: float = 1.0,
    fast_mode: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate dual-stream model on event prediction using walk-forward validation.
    
    Flattens theta stream and concatenates with mellin features to form a 2D matrix
    for LogisticRegression.
    
    Returns:
        aggregate_metrics: Aggregated metrics across folds
        fold_metrics: List of per-fold metrics with sanity checks
    """
    print("\n" + "=" * 70)
    print("DUAL-STREAM MODEL - EVENT PREDICTION")
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
    
    # Build dual-stream features
    print(f"  Building Theta+Mellin features...")
    t_start = time.perf_counter()
    index, X_theta, X_mellin = build_dual_stream_inputs(
        df=df,
        window=theta_window,
        q=theta_q,
        n_terms=theta_terms,
        mellin_k=mellin_k,
        alpha=mellin_alpha,
        omega_max=mellin_omega_max,
    )
    
    # Align with event labels
    common_index = index.intersection(event_labels.index)
    idx_mask = index.isin(common_index)
    X_theta = X_theta[idx_mask]
    X_mellin = X_mellin[idx_mask]
    index = index[idx_mask]
    
    labels = event_labels.loc[index].values
    future_vol_aligned = future_vol.loc[index].values
    
    # Flatten theta stream and concatenate with mellin
    # X_theta: (N, window) -> flatten to (N, window)
    # X_mellin: (N, mellin_k)
    # X_dual: (N, window + mellin_k)
    X_dual = np.concatenate([X_theta, X_mellin], axis=1)
    
    if fast_mode:
        # Cap samples for fast mode
        N_FAST = 400
        n_fast = min(N_FAST, len(X_dual))
        X_dual = X_dual[:n_fast]
        labels = labels[:n_fast]
        future_vol_aligned = future_vol_aligned[:n_fast]
        index = index[:n_fast]
        print(f"  Fast mode: Using first {n_fast} samples (capped at {N_FAST})")
    
    t_features = time.perf_counter() - t_start
    print(f"  Dual-stream matrix shape: {X_dual.shape}")
    print(f"  Feature building time: {t_features:.2f}s")
    
    # Walk-forward validation
    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, purge=0, embargo=0)
    
    all_metrics = []
    all_predictions = []
    
    t_train_start = time.perf_counter()
    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(index), 1):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        
        print(f"  Fold {fold_num}/{n_splits}: train={len(train_idx)}, test={len(test_idx)}")
        
        X_train = X_dual[train_idx]
        y_train = labels[train_idx]
        future_vol_train = future_vol_aligned[train_idx]
        
        X_test = X_dual[test_idx]
        y_test = labels[test_idx]
        future_vol_test = future_vol_aligned[test_idx]
        
        # Compute fold-specific threshold on TRAIN data only
        train_threshold = np.quantile(future_vol_train, quantile)
        
        # Recompute labels with train threshold (fold-safe)
        y_train_foldsafe = (future_vol_train >= train_threshold).astype(int)
        y_test_foldsafe = (future_vol_test >= train_threshold).astype(int)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train LogisticRegression with balanced class weights
        model = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train_scaled, y_train_foldsafe)
        
        # Predict probabilities
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Compute metrics
        roc_auc = roc_auc_score(y_test_foldsafe, probs)
        pr_auc = average_precision_score(y_test_foldsafe, probs)
        brier = brier_score_loss(y_test_foldsafe, probs)
        event_rate = y_test_foldsafe.mean()
        
        # Gating sanity check: compare top 20% vs bottom 20%
        top_20_pct = np.quantile(probs, 0.80)
        bottom_20_pct = np.quantile(probs, 0.20)
        top_mask = probs >= top_20_pct
        bottom_mask = probs <= bottom_20_pct
        
        if top_mask.sum() > 0 and bottom_mask.sum() > 0:
            mean_vol_top = future_vol_test[top_mask].mean()
            mean_vol_bottom = future_vol_test[bottom_mask].mean()
            vol_ratio = mean_vol_top / (mean_vol_bottom + EPSILON)
            vol_diff = mean_vol_top - mean_vol_bottom
        else:
            mean_vol_top = np.nan
            mean_vol_bottom = np.nan
            vol_ratio = np.nan
            vol_diff = np.nan
        
        fold_metrics = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "event_rate": event_rate,
            "mean_vol_top": mean_vol_top,
            "mean_vol_bottom": mean_vol_bottom,
            "vol_ratio": vol_ratio,
            "vol_diff": vol_diff,
        }
        all_metrics.append(fold_metrics)
        
        # Store predictions
        all_predictions.append({
            "probs": probs,
            "y_true": y_test_foldsafe,
            "future_vol": future_vol_test,
        })
    
    t_train = time.perf_counter() - t_train_start
    print(f"  Training time: {t_train:.2f}s")
    print(f"  ✓ Completed {len(all_metrics)} folds")
    
    # Aggregate metrics
    aggregate_metrics = pd.DataFrame(all_metrics).mean(numeric_only=True).to_dict()
    
    return aggregate_metrics, all_metrics


def generate_report(
    df: pd.DataFrame,
    baseline_metrics: Dict,
    dual_stream_metrics: Dict,
    config: Dict,
    output_path: str,
    fast_mode: bool = False,
) -> None:
    """Generate markdown report comparing baseline and dual-stream event prediction."""
    
    report = f"""# Phase-Collapse Event Prediction Evaluation Report

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

## Dataset Summary

- **Symbol:** BTCUSDT
- **Timeframe:** 1 Hour (1H)
- **Dataset Path:** {config['data_path']}
- **Date Range:** {df.index[0]} to {df.index[-1]}
- **Total Bars:** {len(df):,}
- **Price Range:** ${df['close'].min():.2f} to ${df['close'].max():.2f}

## Event Definition

**Event Type:** VOL_BURST

An event occurs when future realized volatility over the next H bars is high (>= quantile threshold).

**Parameters:**
- **Horizon (H):** {config['horizon']} bars
- **Quantile:** {config['quantile']:.2f} (defines high volatility threshold)

**Definition (no leakage):**
```
r[t] = log(close[t]) - log(close[t-1])
future_vol[t] = std(r[t+1 : t+H+1])   # STRICTLY future window
threshold = quantile(future_vol_train, q)  # Computed on TRAIN only per fold
event[t] = 1 if future_vol[t] >= threshold else 0
```

## Configuration

- **Mode:** {config.get('mode', 'default').upper()}
- **Walk-Forward Splits:** {config['n_splits']}

### Dual-Stream Parameters
- **Theta Window:** {config['theta_window']}
- **Theta q:** {config['theta_q']}
- **Theta Terms:** {config['theta_terms']}
- **Mellin k:** {config['mellin_k']}
- **Mellin alpha:** {config['mellin_alpha']}
- **Mellin omega_max:** {config['mellin_omega_max']}

## Results Comparison

### Event Prediction Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **ROC-AUC** | {baseline_metrics['roc_auc']:.4f} | {dual_stream_metrics['roc_auc']:.4f} | {(dual_stream_metrics['roc_auc'] - baseline_metrics['roc_auc']):.4f} |
| **PR-AUC (Average Precision)** | {baseline_metrics['pr_auc']:.4f} | {dual_stream_metrics['pr_auc']:.4f} | {(dual_stream_metrics['pr_auc'] - baseline_metrics['pr_auc']):.4f} |
| **Brier Score** | {baseline_metrics['brier_score']:.4f} | {dual_stream_metrics['brier_score']:.4f} | {(dual_stream_metrics['brier_score'] - baseline_metrics['brier_score']):.4f} |
| **Event Rate** | {baseline_metrics['event_rate']:.2%} | {dual_stream_metrics['event_rate']:.2%} | {(dual_stream_metrics['event_rate'] - baseline_metrics['event_rate']):.2%} |

### Gating Sanity Check

This check verifies that the model predictions are meaningful by comparing realized future volatility
between high-confidence predictions (top 20%) and low-confidence predictions (bottom 20%).

**Baseline Model:**
- Mean future_vol (top 20%): {baseline_metrics['mean_vol_top']:.6f}
- Mean future_vol (bottom 20%): {baseline_metrics['mean_vol_bottom']:.6f}
- **Ratio (top/bottom):** {baseline_metrics['vol_ratio']:.2f}
- **Difference (top - bottom):** {baseline_metrics['vol_diff']:.6f}

**Dual-Stream Model:**
- Mean future_vol (top 20%): {dual_stream_metrics['mean_vol_top']:.6f}
- Mean future_vol (bottom 20%): {dual_stream_metrics['mean_vol_bottom']:.6f}
- **Ratio (top/bottom):** {dual_stream_metrics['vol_ratio']:.2f}
- **Difference (top - bottom):** {dual_stream_metrics['vol_diff']:.6f}

**Interpretation:** If the model is meaningful, the top 20% should have higher future volatility than the bottom 20%.
A ratio > 1.0 and positive difference indicate the model is capturing real signal.

## Conclusion

"""
    
    # Generate conclusion based on results
    better_roc = dual_stream_metrics['roc_auc'] > baseline_metrics['roc_auc']
    better_pr = dual_stream_metrics['pr_auc'] > baseline_metrics['pr_auc']
    better_brier = dual_stream_metrics['brier_score'] < baseline_metrics['brier_score']
    
    dual_sanity_pass = dual_stream_metrics['vol_ratio'] > 1.0
    baseline_sanity_pass = baseline_metrics['vol_ratio'] > 1.0
    
    if better_roc and better_pr:
        conclusion = "The Dual-Stream model outperforms the baseline in both ROC-AUC and PR-AUC, "
    elif better_roc or better_pr:
        conclusion = "The Dual-Stream model shows improvement in some metrics over the baseline, "
    else:
        conclusion = "Both models show comparable event prediction performance, "
    
    if dual_sanity_pass and baseline_sanity_pass:
        conclusion += "and both models pass the gating sanity check (vol_ratio > 1.0), indicating they capture real signal in predicting volatility events. "
    elif dual_sanity_pass:
        conclusion += "and the dual-stream model passes the gating sanity check while baseline does not, suggesting dual-stream features are more informative for event prediction. "
    elif baseline_sanity_pass:
        conclusion += "though baseline passes the gating sanity check while dual-stream does not. "
    else:
        conclusion += "but neither model passes the gating sanity check, suggesting limited predictive power for volatility events. "
    
    if better_roc and better_pr and dual_sanity_pass:
        conclusion += "\n\n**The dual-stream representation (Theta + Mellin) improves phase-collapse event prediction compared to baseline features.**"
    elif better_roc or better_pr:
        conclusion += "\n\nThe dual-stream architecture shows potential for event prediction but the improvement is modest."
    else:
        conclusion += "\n\nThe hypothesis that dual-stream features predict phase-collapse events better than direction is not strongly supported by this evaluation."
    
    report += conclusion
    
    # Add diagnostics
    report += "\n\n## Diagnostics Summary\n\n"
    
    if fast_mode:
        report += "**⚠️  FAST MODE / DIAGNOSTIC ONLY**\n\n"
        report += "This report was generated in FAST mode with reduced samples and folds.\n"
        report += "Results are for diagnostic purposes only and should not be used for final conclusions.\n\n"
    
    report += "### Notes\n\n"
    report += "- Event labels are computed with fold-safe thresholding (quantile computed on train set only per fold)\n"
    report += "- LogisticRegression with class_weight='balanced' is used to handle class imbalance\n"
    report += "- Dual-stream features: flattened theta reconstruction + Mellin magnitudes\n"
    report += "- All metrics are averaged across walk-forward folds\n"
    
    # Add footer
    report += "\n---\n"
    report += "*This evaluation uses offline real-market dataset (data/BTCUSDT_1H_real.csv.gz). "
    report += "Results are for hypothesis verification and research purposes only.*\n"
    
    # Write report
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"\n✓ Report written to: {output_path}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate Phase-Collapse Event Prediction on BTCUSDT data"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: fewer folds, fewer samples",
    )
    parser.add_argument(
        "--event_type",
        type=str,
        default="vol_burst",
        choices=["vol_burst"],
        help="Event type to predict (default: vol_burst)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Horizon for future volatility window (default: 8)",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.80,
        help="Quantile threshold for high volatility events (default: 0.80)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/BTCUSDT_1H_real.csv.gz",
        help="Path to data file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/PHASE_COLLAPSE_EVAL_REPORT.md",
        help="Output report path",
    )
    args = parser.parse_args()
    
    # Configuration
    config = {
        "event_type": args.event_type,
        "horizon": args.horizon,
        "quantile": args.quantile,
        "n_splits": 3 if args.fast else 5,
        "data_path": args.data_path,
        # Dual-stream params
        "theta_window": 48,
        "theta_q": 0.9,
        "theta_terms": 8,
        "mellin_k": 16,
        "mellin_alpha": 0.5,
        "mellin_omega_max": 1.0,
    }
    
    print("=" * 70)
    print("PHASE-COLLAPSE EVENT PREDICTION EVALUATION")
    print("=" * 70)
    print(f"Mode: {'FAST' if args.fast else 'FULL'}")
    print(f"Data: {args.data_path}")
    print(f"Event Type: {config['event_type'].upper()}")
    print(f"Horizon: {config['horizon']} bars")
    print(f"Quantile: {config['quantile']}")
    print(f"Report: {args.output}")
    
    # Load data
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"\nERROR: Data file not found: {data_path}")
        print("The evaluation requires: data/BTCUSDT_1H_real.csv.gz")
        return 1
    
    print(f"\nLoading data from {data_path}...")
    df = load_dataset(str(data_path))
    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Create event labels
    print(f"\nCreating {config['event_type'].upper()} event labels...")
    if config['event_type'] == 'vol_burst':
        event_labels, future_vol = make_vol_burst_labels(
            close=df['close'],
            horizon=config['horizon'],
            quantile=config['quantile'],
        )
    else:
        raise ValueError(f"Unsupported event type: {config['event_type']}")
    
    print(f"  Event labels created: {len(event_labels)} samples")
    print(f"  Event rate: {event_labels.mean():.2%}")
    
    # Evaluate baseline
    baseline_metrics, _ = evaluate_baseline_events(
        df=df,
        event_labels=event_labels,
        future_vol=future_vol,
        n_splits=config['n_splits'],
        quantile=config['quantile'],
        fast_mode=args.fast,
    )
    
    # Evaluate dual-stream
    dual_stream_metrics, _ = evaluate_dual_stream_events(
        df=df,
        event_labels=event_labels,
        future_vol=future_vol,
        n_splits=config['n_splits'],
        quantile=config['quantile'],
        theta_window=config['theta_window'],
        theta_q=config['theta_q'],
        theta_terms=config['theta_terms'],
        mellin_k=config['mellin_k'],
        mellin_alpha=config['mellin_alpha'],
        mellin_omega_max=config['mellin_omega_max'],
        fast_mode=args.fast,
    )
    
    # Generate report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_report(
        df=df,
        baseline_metrics=baseline_metrics,
        dual_stream_metrics=dual_stream_metrics,
        config=config,
        output_path=str(output_path),
        fast_mode=args.fast,
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
