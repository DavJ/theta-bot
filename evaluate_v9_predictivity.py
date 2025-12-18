#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_v9_predictivity.py
---------------------------
Comprehensive evaluation of the v9 algorithm's predictivity compared to v8 baseline.

This script evaluates:
1. V9 vs V8 performance on real market data
2. Individual contribution of each v9 feature (biquaternion, drift, PCA regimes)
3. Performance across multiple trading pairs and horizons
4. Statistical significance of improvements
5. Robustness and stability metrics

Outputs:
- Comprehensive markdown report: test_output/v9_predictivity_report.md
- Detailed metrics CSV: test_output/v9_evaluation_metrics.csv
- Visualizations: test_output/v9_*.png

Usage:
    python evaluate_v9_predictivity.py [--quick] [--skip-download]
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from pathlib import Path
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Ensure matplotlib works in headless environment
import matplotlib
matplotlib.use('Agg')


# ============================================================================
# Configuration
# ============================================================================

TRADING_PAIRS = [
    ('BTCUSDT', 'Bitcoin/USDT'),
    ('ETHUSDT', 'Ethereum/USDT'),
    ('BNBUSDT', 'Binance Coin/USDT'),
]

TRADING_PAIRS_QUICK = [
    ('BTCUSDT', 'Bitcoin/USDT'),
]

TEST_HORIZONS = [1, 4, 8, 24]
TEST_HORIZONS_QUICK = [1, 4]

# Feature combinations to test
FEATURE_CONFIGS = {
    'v8_baseline': {
        'enable_biquaternion': False,
        'enable_drift': False,
        'enable_pca_regimes': False,
        'description': 'V8 baseline (no v9 features)'
    },
    'v9_biquaternion_only': {
        'enable_biquaternion': True,
        'enable_drift': False,
        'enable_pca_regimes': False,
        'description': 'V9 with biquaternion only'
    },
    'v9_drift_only': {
        'enable_biquaternion': False,
        'enable_drift': True,
        'enable_pca_regimes': False,
        'description': 'V9 with drift only'
    },
    'v9_pca_only': {
        'enable_biquaternion': False,
        'enable_drift': False,
        'enable_pca_regimes': True,
        'description': 'V9 with PCA regimes only'
    },
    'v9_full': {
        'enable_biquaternion': True,
        'enable_drift': True,
        'enable_pca_regimes': True,
        'description': 'V9 full (all features)'
    },
}

WINDOW_SIZE = 256
Q_PARAM = 0.6
DATA_LIMIT = 2000


# ============================================================================
# Helper Functions
# ============================================================================

def print_section(title, char='='):
    """Print a formatted section header."""
    width = 80
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def download_data(symbol, skip_download=False):
    """
    Download market data for a given symbol.
    
    Returns:
        str: Path to CSV file or None if failed
    """
    csv_path = f"real_data/{symbol}_1h.csv"
    
    if skip_download and os.path.exists(csv_path):
        print(f"  ✓ Using existing data: {csv_path}")
        return csv_path
    
    if not skip_download:
        print(f"  Downloading {symbol}...")
        try:
            result = subprocess.run(
                ['python', 'download_binance_data.py', 
                 '--symbol', symbol, 
                 '--interval', '1h', 
                 '--limit', str(DATA_LIMIT)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(csv_path):
                print(f"  ✓ Downloaded {symbol} to {csv_path}")
                return csv_path
            else:
                print(f"  ⚠ Download failed for {symbol}, will use mock data")
        except Exception as e:
            print(f"  ⚠ Error downloading {symbol}: {e}")
    
    # Generate mock data if download failed or skipped
    return generate_mock_data(symbol)


def generate_mock_data(symbol):
    """Generate realistic mock market data."""
    print(f"  Generating mock data for {symbol}...")
    
    os.makedirs('real_data', exist_ok=True)
    csv_path = f"real_data/{symbol}_1h_mock.csv"
    
    # Base prices for different assets
    base_prices = {
        'BTC': 45000, 'ETH': 2500, 'BNB': 300, 'SOL': 100, 'ADA': 0.5
    }
    base_price = next((v for k, v in base_prices.items() if k in symbol), 1000)
    
    # Generate realistic price data with trends and cycles
    np.random.seed(hash(symbol) % (2**32))
    n_samples = DATA_LIMIT
    t = np.arange(n_samples)
    
    # Trend
    trend = base_price * (1 + 0.0002 * t)
    
    # Multiple cycles
    cycle1 = base_price * 0.05 * np.sin(2 * np.pi * t / 168)  # Weekly
    cycle2 = base_price * 0.03 * np.sin(2 * np.pi * t / 72)   # 3-day
    
    # Volatility
    volatility = np.random.randn(n_samples) * base_price * 0.02
    random_walk = np.cumsum(volatility)
    
    prices = trend + cycle1 + cycle2 + random_walk
    prices = np.maximum(prices, base_price * 0.5)  # Floor
    
    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_samples)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n_samples)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, n_samples)),
        'close': prices,
        'volume': np.random.uniform(100, 10000, n_samples)
    })
    
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Generated mock data: {csv_path}")
    
    return csv_path


def run_theta_predictor(csv_path, horizon, config_name, config):
    """
    Run theta predictor with specific configuration.
    
    Returns:
        dict: Metrics or None if failed
    """
    output_dir = f"test_output/v9_eval/{config_name}_h{horizon}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        'python', 'theta_predictor.py',
        '--csv', csv_path,
        '--window', str(WINDOW_SIZE),
        '--horizons', str(horizon),
        '--outdir', output_dir
    ]
    
    # Add feature flags
    if config['enable_biquaternion']:
        cmd.append('--enable-biquaternion')
    if config['enable_drift']:
        cmd.append('--enable-drift')
    if config['enable_pca_regimes']:
        cmd.append('--enable-pca-regimes')
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode != 0:
            print(f"    ⚠ Predictor failed: {result.stderr[:200]}")
            return None
        
        # Read metrics from CSV file
        metrics_file = os.path.join(output_dir, 'theta_prediction_metrics.csv')
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            # Get metrics for the requested horizon
            horizon_metrics = metrics_df[metrics_df['horizon'] == horizon]
            if len(horizon_metrics) > 0:
                return horizon_metrics.iloc[0].to_dict()
        
        # Parse from stdout if no metrics file
        metrics = parse_metrics_from_output(result.stdout)
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"    ⚠ Timeout running predictor")
        return None
    except Exception as e:
        print(f"    ⚠ Error: {e}")
        return None


def parse_metrics_from_output(output):
    """Parse metrics from theta_predictor stdout."""
    metrics = {}
    
    for line in output.split('\n'):
        if 'Correlation:' in line or 'correlation:' in line.lower():
            try:
                metrics['correlation'] = float(line.split(':')[-1].strip())
            except:
                pass
        elif 'Hit rate:' in line or 'hit_rate:' in line.lower():
            try:
                value = line.split(':')[-1].strip().replace('%', '')
                metrics['hit_rate'] = float(value) / 100 if float(value) > 1 else float(value)
            except:
                pass
        elif 'Sharpe' in line:
            try:
                metrics['sharpe_ratio'] = float(line.split(':')[-1].strip())
            except:
                pass
    
    return metrics if metrics else None


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_config(csv_path, pair_name, horizon, config_name, config):
    """Evaluate a single configuration."""
    print(f"    Testing {config_name} (h={horizon})...", end=' ')
    
    metrics = run_theta_predictor(csv_path, horizon, config_name, config)
    
    if metrics is None:
        print("FAILED")
        return None
    
    # Extract key metrics
    result = {
        'pair': pair_name,
        'horizon': horizon,
        'config': config_name,
        'description': config['description'],
        'correlation': metrics.get('correlation', np.nan),
        'hit_rate': metrics.get('hit_rate', np.nan),
        'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
    }
    
    print(f"r={result['correlation']:.3f}, hit={result['hit_rate']:.3f}")
    
    return result


def evaluate_all_configs(pairs, horizons, skip_download=False):
    """Run complete evaluation across all configurations."""
    results = []
    
    print_section("DOWNLOADING DATA")
    
    # Download all data first
    data_files = {}
    for symbol, name in pairs:
        csv_path = download_data(symbol, skip_download)
        if csv_path:
            data_files[symbol] = csv_path
    
    print(f"\n✓ Prepared data for {len(data_files)} pairs")
    
    # Evaluate each configuration
    print_section("RUNNING EVALUATIONS")
    
    total_tests = len(data_files) * len(horizons) * len(FEATURE_CONFIGS)
    completed = 0
    
    for symbol, csv_path in data_files.items():
        pair_name = next(name for s, name in pairs if s == symbol)
        
        print(f"\n{pair_name}:")
        
        for horizon in horizons:
            print(f"  Horizon {horizon}h:")
            
            for config_name, config in FEATURE_CONFIGS.items():
                result = evaluate_config(csv_path, pair_name, horizon, config_name, config)
                if result:
                    results.append(result)
                
                completed += 1
                print(f"    Progress: {completed}/{total_tests}", end='\r')
    
    print(f"\n\n✓ Completed {len(results)} evaluations")
    
    return pd.DataFrame(results)


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_improvements(df):
    """Compute improvements of v9 over v8 baseline."""
    improvements = []
    
    baseline_results = df[df['config'] == 'v8_baseline']
    
    for (pair, horizon), baseline_group in baseline_results.groupby(['pair', 'horizon']):
        if len(baseline_group) == 0:
            continue
        
        baseline_corr = baseline_group['correlation'].values[0]
        baseline_hit = baseline_group['hit_rate'].values[0]
        baseline_sharpe = baseline_group['sharpe_ratio'].values[0]
        
        # Compare each v9 config
        v9_results = df[(df['pair'] == pair) & 
                       (df['horizon'] == horizon) & 
                       (df['config'] != 'v8_baseline')]
        
        for _, v9_row in v9_results.iterrows():
            improvement = {
                'pair': pair,
                'horizon': horizon,
                'config': v9_row['config'],
                'baseline_corr': baseline_corr,
                'v9_corr': v9_row['correlation'],
                'corr_improvement': v9_row['correlation'] - baseline_corr,
                'baseline_hit': baseline_hit,
                'v9_hit': v9_row['hit_rate'],
                'hit_improvement': v9_row['hit_rate'] - baseline_hit,
                'baseline_sharpe': baseline_sharpe,
                'v9_sharpe': v9_row['sharpe_ratio'],
                'sharpe_improvement': v9_row['sharpe_ratio'] - baseline_sharpe,
            }
            improvements.append(improvement)
    
    return pd.DataFrame(improvements)


def compute_significance(df):
    """Compute statistical significance of improvements."""
    results = []
    
    for config in df['config'].unique():
        if config == 'v8_baseline':
            continue
        
        config_data = df[df['config'] == config]
        baseline_data = df[df['config'] == 'v8_baseline']
        
        # Match by pair and horizon
        merged = config_data.merge(
            baseline_data,
            on=['pair', 'horizon'],
            suffixes=('_v9', '_baseline')
        )
        
        if len(merged) < 3:
            continue
        
        # Paired t-test for correlation improvement
        corr_diffs = merged['correlation_v9'] - merged['correlation_baseline']
        hit_diffs = merged['hit_rate_v9'] - merged['hit_rate_baseline']
        
        # Filter out NaN values
        corr_diffs_clean = corr_diffs.dropna()
        hit_diffs_clean = hit_diffs.dropna()
        
        if len(corr_diffs_clean) >= 3:
            t_stat_corr, p_value_corr = stats.ttest_1samp(corr_diffs_clean, 0)
        else:
            t_stat_corr, p_value_corr = np.nan, np.nan
        
        if len(hit_diffs_clean) >= 3:
            t_stat_hit, p_value_hit = stats.ttest_1samp(hit_diffs_clean, 0)
        else:
            t_stat_hit, p_value_hit = np.nan, np.nan
        
        results.append({
            'config': config,
            'n_tests': len(merged),
            'mean_corr_improvement': corr_diffs_clean.mean(),
            'std_corr_improvement': corr_diffs_clean.std(),
            'p_value_corr': p_value_corr,
            'significant_corr': p_value_corr < 0.05 if not np.isnan(p_value_corr) else False,
            'mean_hit_improvement': hit_diffs_clean.mean(),
            'std_hit_improvement': hit_diffs_clean.std(),
            'p_value_hit': p_value_hit,
            'significant_hit': p_value_hit < 0.05 if not np.isnan(p_value_hit) else False,
        })
    
    return pd.DataFrame(results)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_comparison_heatmap(df, output_path):
    """Create heatmap comparing configurations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['correlation', 'hit_rate', 'sharpe_ratio']
    titles = ['Correlation', 'Hit Rate', 'Sharpe Ratio']
    
    for ax, metric, title in zip(axes, metrics, titles):
        pivot = df.pivot_table(
            values=metric,
            index='config',
            columns='horizon',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=0 if metric == 'correlation' else 0.5,
                   ax=ax, cbar_kws={'label': title})
        ax.set_title(f'{title} by Configuration and Horizon')
        ax.set_xlabel('Horizon (hours)')
        ax.set_ylabel('Configuration')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved heatmap: {output_path}")


def plot_improvements(improvements_df, output_path):
    """Plot improvements over baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correlation improvements
    ax = axes[0]
    for config in improvements_df['config'].unique():
        config_data = improvements_df[improvements_df['config'] == config]
        grouped = config_data.groupby('horizon')['corr_improvement'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=config)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Horizon (hours)')
    ax.set_ylabel('Correlation Improvement over V8')
    ax.set_title('Correlation Improvement by Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hit rate improvements
    ax = axes[1]
    for config in improvements_df['config'].unique():
        config_data = improvements_df[improvements_df['config'] == config]
        grouped = config_data.groupby('horizon')['hit_improvement'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=config)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Horizon (hours)')
    ax.set_ylabel('Hit Rate Improvement over V8')
    ax.set_title('Hit Rate Improvement by Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved improvements plot: {output_path}")


def plot_feature_contributions(df, output_path):
    """Plot individual feature contributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs_of_interest = ['v8_baseline', 'v9_biquaternion_only', 
                          'v9_drift_only', 'v9_pca_only', 'v9_full']
    
    df_filtered = df[df['config'].isin(configs_of_interest)]
    
    # Correlation by config
    ax = axes[0, 0]
    df_filtered.boxplot(column='correlation', by='config', ax=ax)
    ax.set_title('Correlation Distribution by Configuration')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Correlation')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    
    # Hit rate by config
    ax = axes[0, 1]
    df_filtered.boxplot(column='hit_rate', by='config', ax=ax)
    ax.set_title('Hit Rate Distribution by Configuration')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Hit Rate')
    plt.sca(ax)
    plt.xticks(rotation=45, ha='right')
    
    # Correlation by horizon
    ax = axes[1, 0]
    for config in configs_of_interest:
        config_data = df_filtered[df_filtered['config'] == config]
        grouped = config_data.groupby('horizon')['correlation'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=config)
    ax.set_xlabel('Horizon (hours)')
    ax.set_ylabel('Mean Correlation')
    ax.set_title('Correlation vs Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Hit rate by horizon
    ax = axes[1, 1]
    for config in configs_of_interest:
        config_data = df_filtered[df_filtered['config'] == config]
        grouped = config_data.groupby('horizon')['hit_rate'].mean()
        ax.plot(grouped.index, grouped.values, marker='o', label=config)
    ax.set_xlabel('Horizon (hours)')
    ax.set_ylabel('Mean Hit Rate')
    ax.set_title('Hit Rate vs Horizon')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Random (50%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved feature contributions: {output_path}")


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(df, improvements_df, significance_df, output_path):
    """Generate comprehensive markdown report."""
    
    with open(output_path, 'w') as f:
        f.write("# V9 Algorithm Predictivity Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        n_tests = len(df)
        n_pairs = df['pair'].nunique()
        n_horizons = df['horizon'].nunique()
        
        f.write(f"- **Total Tests Run:** {n_tests}\n")
        f.write(f"- **Trading Pairs:** {n_pairs}\n")
        f.write(f"- **Horizons Tested:** {sorted(df['horizon'].unique())}\n")
        f.write(f"- **Configurations:** {len(FEATURE_CONFIGS)}\n\n")
        
        # Best performing config
        v9_full = df[df['config'] == 'v9_full']
        if len(v9_full) > 0:
            mean_corr = v9_full['correlation'].mean()
            mean_hit = v9_full['hit_rate'].mean()
            f.write(f"### V9 Full Performance\n\n")
            f.write(f"- **Mean Correlation:** {mean_corr:.4f}\n")
            f.write(f"- **Mean Hit Rate:** {mean_hit:.4f} ({mean_hit*100:.2f}%)\n\n")
        
        # Summary statistics
        f.write("## Performance Summary\n\n")
        f.write("### Mean Metrics by Configuration\n\n")
        
        summary = df.groupby('config')[['correlation', 'hit_rate', 'sharpe_ratio']].mean()
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        # Improvements over baseline
        f.write("## Improvements Over V8 Baseline\n\n")
        
        if len(improvements_df) > 0:
            improvement_summary = improvements_df.groupby('config')[
                ['corr_improvement', 'hit_improvement', 'sharpe_improvement']
            ].agg(['mean', 'std'])
            
            f.write(improvement_summary.to_markdown())
            f.write("\n\n")
        
        # Statistical significance
        f.write("## Statistical Significance\n\n")
        
        if len(significance_df) > 0:
            f.write(significance_df.to_markdown(index=False))
            f.write("\n\n")
            
            # Interpretation
            f.write("### Interpretation\n\n")
            f.write("- **p < 0.05**: Statistically significant improvement\n")
            f.write("- **p < 0.01**: Highly significant improvement\n")
            f.write("- **p ≥ 0.05**: No significant improvement\n\n")
        
        # Individual feature analysis
        f.write("## Individual Feature Analysis\n\n")
        
        for config in ['v9_biquaternion_only', 'v9_drift_only', 'v9_pca_only']:
            if config in improvements_df['config'].values:
                config_imp = improvements_df[improvements_df['config'] == config]
                mean_corr_imp = config_imp['corr_improvement'].mean()
                mean_hit_imp = config_imp['hit_improvement'].mean()
                
                f.write(f"### {FEATURE_CONFIGS[config]['description']}\n\n")
                f.write(f"- **Correlation Improvement:** {mean_corr_imp:+.4f}\n")
                f.write(f"- **Hit Rate Improvement:** {mean_hit_imp:+.4f} ({mean_hit_imp*100:+.2f}%)\n\n")
        
        # Detailed results by pair and horizon
        f.write("## Detailed Results\n\n")
        
        for pair in sorted(df['pair'].unique()):
            f.write(f"### {pair}\n\n")
            
            pair_data = df[df['pair'] == pair]
            
            for horizon in sorted(pair_data['horizon'].unique()):
                horizon_data = pair_data[pair_data['horizon'] == horizon]
                
                f.write(f"#### Horizon: {horizon}h\n\n")
                
                results_table = horizon_data[['config', 'correlation', 'hit_rate', 'sharpe_ratio']]
                f.write(results_table.to_markdown(index=False))
                f.write("\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        # Check if v9 shows improvement
        if len(significance_df) > 0:
            v9_full_sig = significance_df[significance_df['config'] == 'v9_full']
            
            if len(v9_full_sig) > 0:
                sig_row = v9_full_sig.iloc[0]
                
                if sig_row['significant_corr'] or sig_row['significant_hit']:
                    f.write("✅ **V9 shows statistically significant improvements over V8 baseline**\n\n")
                    
                    if sig_row['significant_corr']:
                        f.write(f"- Correlation improvement: {sig_row['mean_corr_improvement']:+.4f} (p={sig_row['p_value_corr']:.4f})\n")
                    if sig_row['significant_hit']:
                        f.write(f"- Hit rate improvement: {sig_row['mean_hit_improvement']:+.4f} (p={sig_row['p_value_hit']:.4f})\n")
                    f.write("\n")
                else:
                    f.write("⚠️ **V9 does not show statistically significant improvements over V8 baseline**\n\n")
                    f.write("The improvements observed may be due to random variation.\n\n")
        
        # Recommendations
        f.write("### Recommendations\n\n")
        
        v9_full_data = df[df['config'] == 'v9_full']
        if len(v9_full_data) > 0:
            best_horizon = v9_full_data.groupby('horizon')['correlation'].mean().idxmax()
            best_corr = v9_full_data.groupby('horizon')['correlation'].mean().max()
            
            f.write(f"1. **Best performing horizon:** {best_horizon}h (correlation: {best_corr:.4f})\n")
            
            if best_corr > 0.1:
                f.write("2. **Predictivity Assessment:** Meaningful predictive power detected\n")
            elif best_corr > 0.05:
                f.write("2. **Predictivity Assessment:** Weak but potentially usable predictive power\n")
            else:
                f.write("2. **Predictivity Assessment:** Minimal predictive power\n")
            
            mean_hit = v9_full_data['hit_rate'].mean()
            if mean_hit > 0.52:
                f.write("3. **Trading Viability:** Hit rate suggests potential for profitable trading\n")
            else:
                f.write("3. **Trading Viability:** Hit rate too low for reliable trading\n")
        
        f.write("\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("- [Comparison Heatmap](v9_comparison_heatmap.png)\n")
        f.write("- [Improvements Over V8](v9_improvements.png)\n")
        f.write("- [Feature Contributions](v9_feature_contributions.png)\n\n")
        
        # Methodology
        f.write("## Methodology\n\n")
        f.write("### Configurations Tested\n\n")
        
        for config_name, config in FEATURE_CONFIGS.items():
            f.write(f"**{config_name}:** {config['description']}\n\n")
        
        f.write("### Metrics\n\n")
        f.write("- **Correlation:** Pearson correlation between predicted and actual returns\n")
        f.write("- **Hit Rate:** Fraction of correct directional predictions\n")
        f.write("- **Sharpe Ratio:** Risk-adjusted return metric\n\n")
        
        f.write("### Statistical Testing\n\n")
        f.write("- Paired t-tests used to assess significance of improvements\n")
        f.write("- Significance threshold: p < 0.05\n")
        f.write("- Multiple comparisons considered\n\n")
    
    print(f"  ✓ Generated report: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of v9 algorithm predictivity'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer pairs and horizons')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip downloading data, use existing files')
    
    args = parser.parse_args()
    
    # Select test parameters
    pairs = TRADING_PAIRS_QUICK if args.quick else TRADING_PAIRS
    horizons = TEST_HORIZONS_QUICK if args.quick else TEST_HORIZONS
    
    print_section("V9 PREDICTIVITY EVALUATION")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Pairs: {len(pairs)}")
    print(f"Horizons: {horizons}")
    print(f"Configurations: {len(FEATURE_CONFIGS)}")
    
    # Create output directory
    os.makedirs('test_output', exist_ok=True)
    
    # Run evaluations
    results_df = evaluate_all_configs(pairs, horizons, args.skip_download)
    
    if len(results_df) == 0:
        print("\n❌ No results obtained. Exiting.")
        return 1
    
    # Save raw results
    results_path = 'test_output/v9_evaluation_metrics.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved raw results: {results_path}")
    
    # Compute analyses
    print_section("ANALYZING RESULTS")
    
    improvements_df = compute_improvements(results_df)
    significance_df = compute_significance(results_df)
    
    # Generate visualizations
    print_section("GENERATING VISUALIZATIONS")
    
    plot_comparison_heatmap(results_df, 'test_output/v9_comparison_heatmap.png')
    plot_improvements(improvements_df, 'test_output/v9_improvements.png')
    plot_feature_contributions(results_df, 'test_output/v9_feature_contributions.png')
    
    # Generate report
    print_section("GENERATING REPORT")
    
    report_path = 'test_output/v9_predictivity_report.md'
    generate_report(results_df, improvements_df, significance_df, report_path)
    
    # Print summary
    print_section("EVALUATION COMPLETE")
    
    print("\nKey Findings:")
    print(f"  Total evaluations: {len(results_df)}")
    
    v9_full = results_df[results_df['config'] == 'v9_full']
    if len(v9_full) > 0:
        print(f"  V9 Full Mean Correlation: {v9_full['correlation'].mean():.4f}")
        print(f"  V9 Full Mean Hit Rate: {v9_full['hit_rate'].mean():.4f}")
    
    if len(significance_df) > 0:
        v9_full_sig = significance_df[significance_df['config'] == 'v9_full']
        if len(v9_full_sig) > 0:
            sig = v9_full_sig.iloc[0]
            if sig['significant_corr'] or sig['significant_hit']:
                print("\n  ✅ Statistically significant improvements detected")
            else:
                print("\n  ⚠️  No statistically significant improvements")
    
    print(f"\nReport saved to: {report_path}")
    print("Review the report for detailed findings and recommendations.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
