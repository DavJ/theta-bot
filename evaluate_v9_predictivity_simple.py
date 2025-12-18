#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_v9_predictivity_simple.py
-----------------------------------
Simplified evaluation of v9 algorithm predictivity.

This script compares v9 (with all new features) against v8 baseline
using the theta_predictor directly on real market data.

Usage:
    python evaluate_v9_predictivity_simple.py [--quick]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import subprocess

# Configuration
PAIRS_FULL = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
PAIRS_QUICK = ['BTCUSDT']
HORIZONS_FULL = [1, 4, 8]
HORIZONS_QUICK = [1, 4]
WINDOW = 256


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def generate_mock_data(symbol):
    """Generate realistic mock market data."""
    print(f"  Generating mock data for {symbol}...")
    
    os.makedirs('real_data', exist_ok=True)
    output_path = f"real_data/{symbol}_1h_mock.csv"
    
    if os.path.exists(output_path):
        print(f"  ✓ Mock data already exists: {output_path}")
        return output_path
    
    # Base prices
    base_prices = {'BTC': 45000, 'ETH': 2500, 'BNB': 300}
    base = next((v for k, v in base_prices.items() if k in symbol), 1000)
    
    # Generate price series
    np.random.seed(hash(symbol) % (2**32))
    n = 2000
    t = np.arange(n)
    
    trend = base * (1 + 0.0002 * t)
    cycle1 = base * 0.05 * np.sin(2 * np.pi * t / 168)
    cycle2 = base * 0.03 * np.sin(2 * np.pi * t / 72)
    noise = np.cumsum(np.random.randn(n) * base * 0.02)
    
    prices = np.maximum(trend + cycle1 + cycle2 + noise, base * 0.5)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='h'),
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, n)),
        'close': prices,
        'volume': np.random.uniform(100, 10000, n)
    })
    
    df.to_csv(output_path, index=False)
    print(f"  ✓ Generated: {output_path}")
    return output_path


def run_theta_predictor(csv_path, horizons, config_name, enable_v9=False):
    """
    Run theta predictor and return metrics.
    
    Args:
        csv_path: Path to data CSV
        horizons: List of horizons to test
        config_name: Name of configuration
        enable_v9: If True, enable all v9 features
    
    Returns:
        DataFrame with metrics or None
    """
    outdir = f"test_output/v9_eval/{config_name}"
    os.makedirs(outdir, exist_ok=True)
    
    cmd = [
        'python3', 'theta_predictor.py',
        '--csv', csv_path,
        '--window', str(WINDOW),
        '--horizons'] + [str(h) for h in horizons] + [
        '--outdir', outdir
    ]
    
    if enable_v9:
        cmd.extend([
            '--enable-biquaternion',
            '--enable-drift',
            '--enable-pca-regimes'
        ])
    
    try:
        print(f"    Running {config_name}...", end=' ', flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode != 0:
            print(f"FAILED")
            print(f"    Error: {result.stderr[:200]}")
            return None
        
        # Read metrics
        metrics_file = os.path.join(outdir, 'theta_prediction_metrics.csv')
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            df['config'] = config_name
            print(f"OK")
            return df
        else:
            print(f"NO METRICS")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def run_evaluation(pairs, horizons):
    """Run full evaluation."""
    print_header("PREPARING DATA")
    
    # Ensure data exists
    data_files = {}
    for symbol in pairs:
        csv_path = generate_mock_data(symbol)
        if csv_path:
            data_files[symbol] = csv_path
    
    print(f"\n✓ Prepared {len(data_files)} datasets")
    
    print_header("RUNNING EVALUATIONS")
    
    all_results = []
    
    for symbol, csv_path in data_files.items():
        print(f"\n{symbol}:")
        
        # Test v8 baseline
        v8_metrics = run_theta_predictor(csv_path, horizons, f"{symbol}_v8", enable_v9=False)
        if v8_metrics is not None:
            v8_metrics['pair'] = symbol
            all_results.append(v8_metrics)
        
        # Test v9 full
        v9_metrics = run_theta_predictor(csv_path, horizons, f"{symbol}_v9", enable_v9=True)
        if v9_metrics is not None:
            v9_metrics['pair'] = symbol
            all_results.append(v9_metrics)
    
    if not all_results:
        print("\n❌ No results obtained")
        return None
    
    return pd.concat(all_results, ignore_index=True)


def analyze_results(df):
    """Analyze and compare results."""
    print_header("ANALYSIS")
    
    # Split v8 and v9 results
    v8_results = df[df['config'].str.contains('_v8')]
    v9_results = df[df['config'].str.contains('_v9')]
    
    print("\n### V8 Baseline Performance")
    print(v8_results.groupby('horizon')[['correlation', 'hit_rate', 'sharpe_ratio']].mean().to_string())
    
    print("\n### V9 Full Performance")
    print(v9_results.groupby('horizon')[['correlation', 'hit_rate', 'sharpe_ratio']].mean().to_string())
    
    # Compute improvements
    print("\n### Improvements (V9 - V8)")
    
    improvements = []
    for pair in df['pair'].unique():
        v8_pair = v8_results[v8_results['pair'] == pair]
        v9_pair = v9_results[v9_results['pair'] == pair]
        
        for horizon in df['horizon'].unique():
            v8_h = v8_pair[v8_pair['horizon'] == horizon]
            v9_h = v9_pair[v9_pair['horizon'] == horizon]
            
            if len(v8_h) > 0 and len(v9_h) > 0:
                improvements.append({
                    'pair': pair,
                    'horizon': horizon,
                    'corr_v8': v8_h['correlation'].values[0],
                    'corr_v9': v9_h['correlation'].values[0],
                    'corr_diff': v9_h['correlation'].values[0] - v8_h['correlation'].values[0],
                    'hit_v8': v8_h['hit_rate'].values[0],
                    'hit_v9': v9_h['hit_rate'].values[0],
                    'hit_diff': v9_h['hit_rate'].values[0] - v8_h['hit_rate'].values[0],
                })
    
    imp_df = pd.DataFrame(improvements)
    
    if len(imp_df) > 0:
        summary = imp_df.groupby('horizon')[['corr_diff', 'hit_diff']].agg(['mean', 'std'])
        print(summary.to_string())
        
        # Statistical test
        print("\n### Statistical Significance")
        
        if len(imp_df) >= 3:
            # T-test for correlation improvement
            t_stat_corr, p_val_corr = stats.ttest_1samp(imp_df['corr_diff'].dropna(), 0)
            print(f"Correlation improvement: mean={imp_df['corr_diff'].mean():.4f}, p={p_val_corr:.4f}")
            if p_val_corr < 0.05:
                print("  ✅ Statistically significant (p < 0.05)")
            else:
                print("  ⚠️  Not statistically significant")
            
            # T-test for hit rate improvement
            t_stat_hit, p_val_hit = stats.ttest_1samp(imp_df['hit_diff'].dropna(), 0)
            print(f"Hit rate improvement: mean={imp_df['hit_diff'].mean():.4f}, p={p_val_hit:.4f}")
            if p_val_hit < 0.05:
                print("  ✅ Statistically significant (p < 0.05)")
            else:
                print("  ⚠️  Not statistically significant")
        else:
            print("Not enough data points for statistical testing")
        
        return imp_df
    
    return None


def generate_plots(df, imp_df):
    """Generate comparison plots."""
    print_header("GENERATING PLOTS")
    
    os.makedirs('test_output', exist_ok=True)
    
    # Plot 1: Comparison by horizon
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    v8 = df[df['config'].str.contains('_v8')].groupby('horizon')[['correlation', 'hit_rate']].mean()
    v9 = df[df['config'].str.contains('_v9')].groupby('horizon')[['correlation', 'hit_rate']].mean()
    
    # Correlation
    ax = axes[0]
    ax.plot(v8.index, v8['correlation'], 'o-', label='V8 Baseline', linewidth=2)
    ax.plot(v9.index, v9['correlation'], 's-', label='V9 Full', linewidth=2)
    ax.set_xlabel('Horizon (hours)')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation: V9 vs V8')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Hit rate
    ax = axes[1]
    ax.plot(v8.index, v8['hit_rate'], 'o-', label='V8 Baseline', linewidth=2)
    ax.plot(v9.index, v9['hit_rate'], 's-', label='V9 Full', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random (50%)')
    ax.set_xlabel('Horizon (hours)')
    ax.set_ylabel('Hit Rate')
    ax.set_title('Hit Rate: V9 vs V8')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_output/v9_vs_v8_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: test_output/v9_vs_v8_comparison.png")
    
    # Plot 2: Improvements
    if imp_df is not None and len(imp_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Correlation improvement
        ax = axes[0]
        horizons = sorted(imp_df['horizon'].unique())
        for horizon in horizons:
            h_data = imp_df[imp_df['horizon'] == horizon]['corr_diff']
            ax.scatter([horizon] * len(h_data), h_data, alpha=0.6)
        
        grouped = imp_df.groupby('horizon')['corr_diff'].mean()
        ax.plot(grouped.index, grouped.values, 'r-', linewidth=2, label='Mean')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Horizon (hours)')
        ax.set_ylabel('Correlation Improvement (V9 - V8)')
        ax.set_title('Correlation Improvement by Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Hit rate improvement
        ax = axes[1]
        for horizon in horizons:
            h_data = imp_df[imp_df['horizon'] == horizon]['hit_diff']
            ax.scatter([horizon] * len(h_data), h_data, alpha=0.6)
        
        grouped = imp_df.groupby('horizon')['hit_diff'].mean()
        ax.plot(grouped.index, grouped.values, 'r-', linewidth=2, label='Mean')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Horizon (hours)')
        ax.set_ylabel('Hit Rate Improvement (V9 - V8)')
        ax.set_title('Hit Rate Improvement by Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_output/v9_improvements.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: test_output/v9_improvements.png")


def generate_report(df, imp_df):
    """Generate markdown report."""
    print_header("GENERATING REPORT")
    
    report_path = 'test_output/v9_predictivity_evaluation.md'
    
    with open(report_path, 'w') as f:
        f.write("# V9 Algorithm Predictivity Evaluation\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Tests:** {len(df)}\n")
        f.write(f"- **Pairs Tested:** {df['pair'].nunique()}\n")
        f.write(f"- **Horizons:** {sorted(df['horizon'].unique())}\n\n")
        
        # V8 vs V9 comparison
        f.write("## Performance Comparison\n\n")
        
        f.write("### V8 Baseline\n\n")
        v8 = df[df['config'].str.contains('_v8')].groupby('horizon')[['correlation', 'hit_rate', 'sharpe_ratio']].mean()
        f.write(v8.to_markdown())
        f.write("\n\n")
        
        f.write("### V9 Full (All Features)\n\n")
        v9 = df[df['config'].str.contains('_v9')].groupby('horizon')[['correlation', 'hit_rate', 'sharpe_ratio']].mean()
        f.write(v9.to_markdown())
        f.write("\n\n")
        
        # Improvements
        if imp_df is not None and len(imp_df) > 0:
            f.write("## Improvements (V9 - V8)\n\n")
            summary = imp_df.groupby('horizon')[['corr_diff', 'hit_diff']].agg(['mean', 'std'])
            f.write(summary.to_markdown())
            f.write("\n\n")
            
            # Statistical tests
            if len(imp_df) >= 3:
                t_stat_corr, p_val_corr = stats.ttest_1samp(imp_df['corr_diff'].dropna(), 0)
                t_stat_hit, p_val_hit = stats.ttest_1samp(imp_df['hit_diff'].dropna(), 0)
                
                f.write("## Statistical Significance\n\n")
                f.write(f"- **Correlation improvement:** mean = {imp_df['corr_diff'].mean():+.4f}, p = {p_val_corr:.4f}\n")
                if p_val_corr < 0.05:
                    f.write("  - ✅ Statistically significant\n")
                else:
                    f.write("  - ⚠️ Not statistically significant\n")
                
                f.write(f"- **Hit rate improvement:** mean = {imp_df['hit_diff'].mean():+.4f}, p = {p_val_hit:.4f}\n")
                if p_val_hit < 0.05:
                    f.write("  - ✅ Statistically significant\n")
                else:
                    f.write("  - ⚠️ Not statistically significant\n")
                f.write("\n")
        
        # Detailed results
        f.write("## Detailed Results\n\n")
        for pair in sorted(df['pair'].unique()):
            f.write(f"### {pair}\n\n")
            pair_data = df[df['pair'] == pair][['config', 'horizon', 'correlation', 'hit_rate', 'sharpe_ratio']]
            f.write(pair_data.to_markdown(index=False))
            f.write("\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        v9_mean_corr = v9['correlation'].mean()
        v9_mean_hit = v9['hit_rate'].mean()
        
        f.write(f"### V9 Performance Summary\n\n")
        f.write(f"- **Mean Correlation:** {v9_mean_corr:.4f}\n")
        f.write(f"- **Mean Hit Rate:** {v9_mean_hit:.4f} ({v9_mean_hit*100:.2f}%)\n\n")
        
        if v9_mean_corr > 0.05:
            f.write("✅ **V9 shows meaningful predictive power** (correlation > 0.05)\n\n")
        else:
            f.write("⚠️ **V9 shows weak predictive power** (correlation < 0.05)\n\n")
        
        if v9_mean_hit > 0.52:
            f.write("✅ **V9 hit rate above random** (> 52%)\n\n")
        else:
            f.write("⚠️ **V9 hit rate near random** (< 52%)\n\n")
        
        f.write("### Visualizations\n\n")
        f.write("- [V9 vs V8 Comparison](v9_vs_v8_comparison.png)\n")
        f.write("- [Improvements](v9_improvements.png)\n\n")
        
        f.write("### Methodology\n\n")
        f.write("- **V8 Baseline:** Standard theta predictor without v9 features\n")
        f.write("- **V9 Full:** Theta predictor with all v9 features enabled:\n")
        f.write("  - Biquaternion time support\n")
        f.write("  - Fokker-Planck drift term\n")
        f.write("  - PCA regime detection\n")
        f.write("- **Window Size:** 256\n")
        f.write("- **Evaluation:** Walk-forward validation (no data leaks)\n\n")
    
    print(f"  ✓ Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate v9 algorithm predictivity')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer pairs)')
    args = parser.parse_args()
    
    pairs = PAIRS_QUICK if args.quick else PAIRS_FULL
    horizons = HORIZONS_QUICK if args.quick else HORIZONS_FULL
    
    print_header("V9 PREDICTIVITY EVALUATION")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Pairs: {pairs}")
    print(f"Horizons: {horizons}")
    
    # Run evaluation
    results_df = run_evaluation(pairs, horizons)
    
    if results_df is None:
        print("\n❌ Evaluation failed")
        return 1
    
    # Save raw results
    results_df.to_csv('test_output/v9_evaluation_results.csv', index=False)
    print(f"\n✓ Saved raw results: test_output/v9_evaluation_results.csv")
    
    # Analyze
    imp_df = analyze_results(results_df)
    
    if imp_df is not None:
        imp_df.to_csv('test_output/v9_improvements.csv', index=False)
        print(f"\n✓ Saved improvements: test_output/v9_improvements.csv")
    
    # Generate plots
    generate_plots(results_df, imp_df)
    
    # Generate report
    generate_report(results_df, imp_df)
    
    print_header("EVALUATION COMPLETE")
    print(f"\n📊 Report: test_output/v9_predictivity_evaluation.md")
    print(f"📈 Plots: test_output/v9_*.png")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
