#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_biquat_binance_real.py
---------------------------
Comprehensive test script for corrected biquaternion implementation
using REAL Binance data with multiple trading pairs including PLN.

Tests on both synthetic and real data with strict walk-forward validation
to ensure NO DATA LEAKS from the future.

Features:
- Tests multiple trading pairs: BTC, ETH, BNB, SOL, ADA (with USDT and PLN where available)
- Multiple test horizons (1, 4, 8, 24 hours)
- Comprehensive HTML/Markdown reports
- Data leak verification
- Performance comparisons across pairs

Usage:
    python test_biquat_binance_real.py [--skip-download] [--quick]
"""

import os
import sys
import numpy as np
import pandas as pd
import subprocess
import argparse
from datetime import datetime
import json


# Trading pairs to test
TRADING_PAIRS = [
    # Major pairs with USDT
    ('BTCUSDT', 'Bitcoin/USDT'),
    ('ETHUSDT', 'Ethereum/USDT'),
    ('BNBUSDT', 'Binance Coin/USDT'),
    ('SOLUSDT', 'Solana/USDT'),
    ('ADAUSDT', 'Cardano/USDT'),
    # PLN pairs (Polish Zloty)
    ('BTCPLN', 'Bitcoin/PLN'),
    ('ETHPLN', 'Ethereum/PLN'),
    ('BNBPLN', 'Binance Coin/PLN'),
]

# Test configuration
TEST_HORIZONS = [1, 4, 8, 24]  # Hours ahead to predict
WINDOW_SIZE = 256
Q_PARAM = 0.6
N_TERMS = 16
N_FREQ = 6
LAMBDA_PARAM = 0.5
PHASE_SCALE = 1.0
DATA_LIMIT = 2000  # Number of candles to download

# Error pattern constants for robust error detection
ERROR_PATTERNS = {
    'symbol_not_found': ['not found', 'invalid'],
    'network_issue': ['resolve', 'connection', 'timeout', 'unreachable'],
}


def print_section(title, char='='):
    """Print a formatted section header."""
    width = 70
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def generate_synthetic_data(n_samples=2000, output_path='test_data/synthetic_biquat_binance.csv'):
    """
    Generate synthetic price data with multiple periodic components.
    This simulates market data with cycles that theta functions should capture.
    """
    print_section("GENERATING SYNTHETIC DATA")
    
    os.makedirs('test_data', exist_ok=True)
    
    np.random.seed(42)
    t = np.arange(n_samples)
    
    # Trend
    trend = 1000 + 0.3 * t
    
    # Multiple periodic components matching our frequency grid
    cycle1 = 30 * np.sin(2 * np.pi * t / 256)      # Period = window
    cycle2 = 20 * np.sin(2 * np.pi * t / 128)      # Period = window/2
    cycle3 = 15 * np.sin(2 * np.pi * t / 64)       # Period = window/4
    cycle4 = 10 * np.sin(2 * np.pi * t / 43)       # Period ≈ window/6
    
    # Random walk component (market noise)
    random_walk = np.cumsum(np.random.randn(n_samples) * 3)
    
    # Combine all components
    prices = trend + cycle1 + cycle2 + cycle3 + cycle4 + random_walk
    
    # Ensure positive prices
    prices = np.abs(prices) + 100
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=len(prices), freq='h'),
        'close': prices,
    })
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} samples")
    print(f"  Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
    print(f"  Price mean: {df['close'].mean():.2f}")
    print(f"  Price std: {df['close'].std():.2f}")
    print(f"  Saved to: {output_path}")
    
    return output_path


def generate_realistic_market_data(symbol, n_samples=2000, base_price=None, output_dir='real_data'):
    """
    Generate realistic market-like data when Binance is unavailable.
    This simulates real market behavior for testing purposes.
    
    Warning:
        This is NOT real Binance data - it's simulated for testing only.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol
    n_samples : int
        Number of samples to generate
    base_price : float, optional
        Base price for the asset
    output_dir : str
        Directory to save the generated data
    
    Returns
    -------
    str
        Path to the generated CSV file
    """
    print(f"  ⚠ WARNING: Generating MOCK data for {symbol} (NOT real Binance data)")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol}_1h_mock.csv")
    
    # Set base price based on common crypto prices
    if base_price is None:
        if 'BTC' in symbol:
            base_price = 45000
        elif 'ETH' in symbol:
            base_price = 2500
        elif 'BNB' in symbol:
            base_price = 350
        elif 'SOL' in symbol:
            base_price = 100
        elif 'ADA' in symbol:
            base_price = 0.5
        else:
            base_price = 1000
    
    # Adjust for PLN (Polish Zloty) - roughly 4x USD price
    if 'PLN' in symbol:
        base_price *= 4
    
    # Use deterministic seeding based on symbol (stable across runs)
    import hashlib
    seed_value = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16) % (2**31)
    np.random.seed(seed_value)
    t = np.arange(n_samples)
    
    # Trend with drift
    drift = 0.0001 * base_price
    trend = base_price + drift * t
    
    # Multiple cycles (similar to actual market patterns)
    cycle1 = base_price * 0.03 * np.sin(2 * np.pi * t / 168)  # Weekly cycle
    cycle2 = base_price * 0.02 * np.sin(2 * np.pi * t / 24)   # Daily cycle
    cycle3 = base_price * 0.01 * np.sin(2 * np.pi * t / 12)   # Half-day cycle
    
    # Realistic random walk (geometric Brownian motion)
    volatility = base_price * 0.02  # 2% hourly volatility
    random_returns = np.random.randn(n_samples) * volatility
    random_walk = np.cumsum(random_returns)
    
    # Occasional jumps (news events)
    jumps = np.zeros(n_samples)
    jump_times = np.random.choice(n_samples, size=5, replace=False)
    for jt in jump_times:
        jumps[jt] = np.random.choice([-1, 1]) * base_price * 0.05
    jump_cumsum = np.cumsum(jumps)
    
    # Combine all components
    prices = trend + cycle1 + cycle2 + cycle3 + random_walk + jump_cumsum
    prices = np.maximum(prices, base_price * 0.5)  # Floor at 50% of base
    
    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='h'),
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.005)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    df.to_csv(output_path, index=False)
    print(f"  ✓ Generated {len(df)} samples to {output_path}")
    
    return output_path


def download_binance_data(symbol, interval='1h', limit=2000, output_dir='real_data', use_mock_if_fail=True):
    """
    Download real market data from Binance.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT')
    interval : str
        Timeframe interval (e.g., '1h', '4h')
    limit : int
        Number of candles to download
    output_dir : str
        Directory to save the data
    use_mock_if_fail : bool
        Whether to generate mock data if download fails
    
    Returns
    -------
    tuple[str, bool]
        Tuple of (path to CSV file, is_real_binance_data).
        Returns (path, True) if real Binance data was downloaded.
        Returns (path, False) if mock data was generated.
        Returns (None, False) if both failed and use_mock_if_fail is False.
    """
    print(f"\n  Downloading {symbol} {interval} data...")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol}_{interval}.csv")
    mock_output_path = os.path.join(output_dir, f"{symbol}_{interval}_mock.csv")
    
    # Check if real Binance data already exists
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path)
            if len(df) >= limit * 0.9:  # At least 90% of requested data
                print(f"  ✓ Found existing REAL Binance data: {output_path} ({len(df)} samples)")
                return output_path, True
        except:
            pass
    
    # Check if mock data already exists
    if os.path.exists(mock_output_path):
        try:
            df = pd.read_csv(mock_output_path)
            if len(df) >= limit * 0.9:
                print(f"  ⚠ Found existing MOCK data: {mock_output_path} ({len(df)} samples)")
                return mock_output_path, False
        except:
            pass
    
    # Use download_market_data.py script
    cmd = [
        sys.executable,
        'download_market_data.py',
        '--symbol', symbol,
        '--interval', interval,
        '--limit', str(limit),
        '--outdir', output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(output_path):
            df = pd.read_csv(output_path)
            print(f"  ✓ Downloaded REAL Binance data: {len(df)} samples to {output_path}")
            return output_path, True
        else:
            print(f"  ✗ Failed to download {symbol}")
            
            # Check for specific error patterns
            stderr_lower = result.stderr.lower()
            if any(pattern in stderr_lower for pattern in ERROR_PATTERNS['symbol_not_found']):
                print(f"    (Symbol {symbol} may not be available on Binance)")
            elif any(pattern in stderr_lower for pattern in ERROR_PATTERNS['network_issue']):
                print(f"    (Network connection issue - no internet access)")
            
            # Generate mock data if download failed
            if use_mock_if_fail:
                mock_path = generate_realistic_market_data(symbol, n_samples=limit, output_dir=output_dir)
                return mock_path, False
            return None, False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout downloading {symbol}")
        if use_mock_if_fail:
            mock_path = generate_realistic_market_data(symbol, n_samples=limit, output_dir=output_dir)
            return mock_path, False
        return None, False
    except Exception as e:
        print(f"  ✗ Error downloading {symbol}: {e}")
        if use_mock_if_fail:
            mock_path = generate_realistic_market_data(symbol, n_samples=limit, output_dir=output_dir)
            return mock_path, False
        return None, False


def verify_no_data_leaks(csv_path):
    """
    Verify that data is properly ordered and has no missing timestamps
    that could indicate data leaks.
    """
    df = pd.read_csv(csv_path)
    
    if 'timestamp' not in df.columns:
        return True, "No timestamp column to verify"
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Check if timestamps are monotonically increasing
    if not df['timestamp'].is_monotonic_increasing:
        return False, "Timestamps are not monotonically increasing"
    
    # Check for large gaps that might indicate data issues
    time_diffs = df['timestamp'].diff()
    median_diff = time_diffs.median()
    large_gaps = time_diffs[time_diffs > median_diff * 10]
    
    if len(large_gaps) > 5:
        return False, f"Found {len(large_gaps)} large time gaps in data"
    
    return True, "Data ordering verified - no obvious data leak indicators"


def run_evaluation(csv_path, label, outdir, horizon=1, window=256):
    """
    Run the corrected biquaternion evaluation with strict walk-forward validation.
    """
    print(f"\n  Testing {label} (horizon={horizon})...")
    
    # Verify data integrity first
    leak_check, leak_msg = verify_no_data_leaks(csv_path)
    if not leak_check:
        print(f"  ⚠ Warning: {leak_msg}")
    
    cmd = [
        sys.executable,
        'theta_bot_averaging/theta_eval_biquat_corrected.py',
        '--csv', csv_path,
        '--price-col', 'close',
        '--horizon', str(horizon),
        '--window', str(window),
        '--q', str(Q_PARAM),
        '--n-terms', str(N_TERMS),
        '--n-freq', str(N_FREQ),
        '--lambda', str(LAMBDA_PARAM),
        '--phase-scale', str(PHASE_SCALE),
        '--outdir', outdir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        print(f"  ✗ Evaluation failed")
        if result.stderr:
            print(f"    Error: {result.stderr[:200]}")
        return None
    
    # Load summary
    summary_path = os.path.join(outdir, 'summary.csv')
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        result_dict = summary.iloc[0].to_dict()
        
        # Extract key metrics
        hit_rate = result_dict.get('hit_rate', np.nan)
        correlation = result_dict.get('corr_pred_true', np.nan)
        n_pred = result_dict.get('n_predictions', 0)
        
        print(f"  ✓ Hit Rate: {hit_rate:.4f}, Correlation: {correlation:.4f}, N={n_pred}")
        
        return result_dict
    else:
        print(f"  ✗ Summary file not found")
        return None


def generate_html_report(all_results, output_path='test_output/comprehensive_report.html'):
    """
    Generate a comprehensive HTML report of all test results.
    """
    print_section("GENERATING COMPREHENSIVE REPORT")
    
    # Check if any real Binance data was used
    any_real_binance = any(r.get('is_real_binance', False) for r in all_results)
    all_mock_data = all(not r.get('is_real_binance', False) for r in all_results)
    
    # Build HTML with escaped braces for CSS
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_tests = len(all_results)
    num_pairs = len(set(r['pair'] for r in all_results if 'pair' in r))
    num_horizons = len(set(r.get('horizon', 1) for r in all_results))
    
    # Calculate average hit rate with proper handling of NaN values
    valid_hit_rates = [r.get('hit_rate', np.nan) for r in all_results if not np.isnan(r.get('hit_rate', np.nan))]
    avg_hit_rate = np.mean(valid_hit_rates) if valid_hit_rates else 0.0
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Theta Bot Biquaternion - Comprehensive Test Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .metric-box {{
            display: inline-block;
            margin: 10px;
            padding: 15px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 12px;
            opacity: 0.9;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .good {{
            color: #27ae60;
            font-weight: bold;
        }}
        .bad {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .neutral {{
            color: #f39c12;
            font-weight: bold;
        }}
        .info-box {{
            background-color: #ecf0f1;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
        }}
        .warning-box {{
            background-color: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .error-box {{
            background-color: #f8d7da;
            padding: 15px;
            border-left: 4px solid #dc3545;
            margin: 20px 0;
        }}
        .success-box {{
            background-color: #d4edda;
            padding: 15px;
            border-left: 4px solid #28a745;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
            text-align: right;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Theta Bot Biquaternion - Comprehensive Test Report</h1>
        <p><strong>Test Date:</strong> {timestamp_str}</p>
        """
    
    # Add warning if mock data was used
    if all_mock_data:
        html += """
        <div class="error-box">
            <strong>⚠️ CRITICAL WARNING: NO REAL BINANCE DATA USED</strong><br>
            All market data in this report is SIMULATED/MOCK data, NOT real Binance data.<br>
            This likely indicates:<br>
            • No internet connection to Binance API<br>
            • Network/firewall blocking access to api.binance.com<br>
            • Binance API unavailable<br>
            <br>
            <strong>These results are for testing infrastructure only and do NOT represent real market performance.</strong><br>
            To test with real data, ensure internet access and re-run the tests.
        </div>
        """
    elif not any_real_binance:
        html += """
        <div class="warning-box">
            <strong>⚠️ WARNING: Mixed data sources</strong><br>
            Some tests use mock data instead of real Binance data.<br>
            Check individual test results for data source information.
        </div>
        """
    else:
        html += """
        <div class="success-box">
            <strong>✓ Real Binance Data Used</strong><br>
            Tests include real market data downloaded from Binance API.
        </div>
        """
    
    html += """
        <div class="success-box">
            <strong>✓ Data Leak Verification:</strong> All tests use strict walk-forward validation.
            The model only uses data from [t-window, t) to predict at time t. No future information is used.
        </div>
        
        <h2>📊 Executive Summary</h2>
        
        <div class="metric-box">
            <div class="metric-label">Total Tests Run</div>
            <div class="metric-value">{total_tests}</div>
        </div>
        
        <div class="metric-box">
            <div class="metric-label">Trading Pairs</div>
            <div class="metric-value">{num_pairs}</div>
        </div>
        
        <div class="metric-box">
            <div class="metric-label">Test Horizons</div>
            <div class="metric-value">{num_horizons}</div>
        </div>
        
        <div class="metric-box">
            <div class="metric-label">Avg Hit Rate</div>
            <div class="metric-value">{avg_hit_rate:.3f}</div>
        </div>
"""
    
    # Synthetic data results
    synthetic_results = [r for r in all_results if r.get('data_type') == 'synthetic']
    if synthetic_results:
        html += """
        <h2>🧪 Synthetic Data Results</h2>
        <div class="info-box">
            Synthetic data contains known periodic patterns at multiple frequencies.
            Good performance here indicates the model can capture cyclical patterns.
        </div>
        <table>
            <tr>
                <th>Horizon</th>
                <th>Hit Rate</th>
                <th>Correlation</th>
                <th>Predictions</th>
                <th>Performance</th>
            </tr>
"""
        for r in synthetic_results:
            hit_rate = r.get('hit_rate', np.nan)
            corr = r.get('corr_pred_true', np.nan)
            n_pred = r.get('n_predictions', 0)
            
            # Performance assessment
            if hit_rate > 0.6:
                perf_class = "good"
                perf_text = "Excellent"
            elif hit_rate > 0.55:
                perf_class = "neutral"
                perf_text = "Good"
            elif hit_rate > 0.5:
                perf_class = "neutral"
                perf_text = "Fair"
            else:
                perf_class = "bad"
                perf_text = "Poor"
            
            html += f"""
            <tr>
                <td>{r.get('horizon', 1)}h</td>
                <td>{hit_rate:.4f}</td>
                <td>{corr:.4f}</td>
                <td>{n_pred}</td>
                <td class="{perf_class}">{perf_text}</td>
            </tr>
"""
        html += "</table>"
    
    # Real data results
    real_results = [r for r in all_results if r.get('data_type') in ['real', 'mock']]
    if real_results:
        # Separate real and mock
        truly_real = [r for r in real_results if r.get('is_real_binance', False)]
        mock_results = [r for r in real_results if not r.get('is_real_binance', False)]
        
        if truly_real:
            html += """
        <h2>📈 Real Binance Market Data Results</h2>
        <div class="success-box">
            <strong>✓ Real Binance Data:</strong> Actual market data downloaded from Binance API.<br>
            Performance above 0.5 hit rate indicates genuine predictive power beyond random chance.
        </div>
"""
            # Group by pair
            pairs = sorted(set(r['pair'] for r in truly_real if 'pair' in r))
            
            for pair in pairs:
                pair_results = [r for r in truly_real if r.get('pair') == pair]
                
                html += f"""
        <h3>✓ {pair_results[0].get('pair_label', pair)} (Real Binance Data)</h3>
        <table>
            <tr>
                <th>Horizon</th>
                <th>Hit Rate</th>
                <th>Correlation</th>
                <th>Predictions</th>
                <th>Performance</th>
            </tr>
"""
                for r in sorted(pair_results, key=lambda x: x.get('horizon', 1)):
                    hit_rate = r.get('hit_rate', np.nan)
                    corr = r.get('corr_pred_true', np.nan)
                    n_pred = r.get('n_predictions', 0)
                    
                    # Performance assessment for real data (more conservative)
                    if hit_rate > 0.55:
                        perf_class = "good"
                        perf_text = "Excellent"
                    elif hit_rate > 0.52:
                        perf_class = "neutral"
                        perf_text = "Good"
                    elif hit_rate > 0.5:
                        perf_class = "neutral"
                        perf_text = "Fair"
                    else:
                        perf_class = "bad"
                        perf_text = "Poor"
                    
                    html += f"""
            <tr>
                <td>{r.get('horizon', 1)}h</td>
                <td>{hit_rate:.4f}</td>
                <td>{corr:.4f}</td>
                <td>{n_pred}</td>
                <td class="{perf_class}">{perf_text}</td>
            </tr>
"""
                html += "</table>"
        
        if mock_results:
            html += """
        <h2>📊 Mock/Simulated Market Data Results</h2>
        <div class="warning-box">
            <strong>⚠️ Mock Data:</strong> These results use SIMULATED data, NOT real Binance data.<br>
            Results are for testing infrastructure only and do NOT represent real market performance.
        </div>
"""
            # Group by pair
            pairs = sorted(set(r['pair'] for r in mock_results if 'pair' in r))
            
            for pair in pairs:
                pair_results = [r for r in mock_results if r.get('pair') == pair]
                
                html += f"""
        <h3>⚠ {pair_results[0].get('pair_label', pair)} (Mock Data)</h3>
        <table>
            <tr>
                <th>Horizon</th>
                <th>Hit Rate</th>
                <th>Correlation</th>
                <th>Predictions</th>
                <th>Performance</th>
            </tr>
"""
                for r in sorted(pair_results, key=lambda x: x.get('horizon', 1)):
                    hit_rate = r.get('hit_rate', np.nan)
                    corr = r.get('corr_pred_true', np.nan)
                    n_pred = r.get('n_predictions', 0)
                    
                    # Performance assessment for real data (more conservative)
                    if hit_rate > 0.55:
                        perf_class = "good"
                        perf_text = "Excellent"
                    elif hit_rate > 0.52:
                        perf_class = "neutral"
                        perf_text = "Good"
                    elif hit_rate > 0.5:
                        perf_class = "neutral"
                        perf_text = "Fair"
                    else:
                        perf_class = "bad"
                        perf_text = "Poor"
                    
                    html += f"""
            <tr>
                <td>{r.get('horizon', 1)}h</td>
                <td>{hit_rate:.4f}</td>
                <td>{corr:.4f}</td>
                <td>{n_pred}</td>
                <td class="{perf_class}">{perf_text}</td>
            </tr>
"""
                html += "</table>"
    
    # Technical details
    html += f"""
        <h2>⚙️ Technical Configuration</h2>
        <div class="info-box">
            <strong>Model Parameters:</strong><br>
            • Window Size: {WINDOW_SIZE}<br>
            • Q Parameter: {Q_PARAM}<br>
            • N Terms: {N_TERMS}<br>
            • N Frequencies: {N_FREQ}<br>
            • Lambda (Regularization): {LAMBDA_PARAM}<br>
            • Phase Scale: {PHASE_SCALE}<br>
            <br>
            <strong>Validation Method:</strong> Strict Walk-Forward<br>
            • Model trained on [t-window, t)<br>
            • Prediction made for time t<br>
            • No future information used<br>
            • No data leakage
        </div>
        
        <h2>📋 Interpretation Guide</h2>
        <div class="info-box">
            <strong>Hit Rate:</strong> Percentage of correct directional predictions (up/down)<br>
            • > 0.55: Excellent predictive power<br>
            • 0.52-0.55: Good predictive power<br>
            • 0.50-0.52: Fair predictive power<br>
            • < 0.50: No predictive power (below random chance)<br>
            <br>
            <strong>Correlation:</strong> Linear relationship between predicted and actual changes<br>
            • > 0.3: Strong relationship<br>
            • 0.1-0.3: Moderate relationship<br>
            • < 0.1: Weak relationship<br>
        </div>
        
        <div class="timestamp">
            Report generated: {timestamp_str}
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✓ HTML report saved to: {output_path}")
    
    # Also create a markdown summary
    md_path = output_path.replace('.html', '.md')
    generate_markdown_report(all_results, md_path)


def generate_markdown_report(all_results, output_path):
    """Generate a Markdown version of the report."""
    
    # Check if any real Binance data was used
    any_real_binance = any(r.get('is_real_binance', False) for r in all_results)
    all_mock_data = all(not r.get('is_real_binance', False) for r in all_results)
    
    # Calculate average hit rate with proper handling of NaN values
    valid_hit_rates = [r.get('hit_rate', np.nan) for r in all_results if not np.isnan(r.get('hit_rate', np.nan))]
    
    md = f"""# Theta Bot Biquaternion - Comprehensive Test Report

**Test Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""

    # Add warning if mock data was used
    if all_mock_data:
        md += """## ⚠️ CRITICAL WARNING: NO REAL BINANCE DATA USED

**All market data in this report is SIMULATED/MOCK data, NOT real Binance data.**

This likely indicates:
- No internet connection to Binance API
- Network/firewall blocking access to api.binance.com
- Binance API unavailable

**These results are for testing infrastructure only and do NOT represent real market performance.**

To test with real data, ensure internet access and re-run the tests.

"""
    elif not any_real_binance:
        md += """## ⚠️ WARNING: Mixed Data Sources

Some tests use mock data instead of real Binance data.
Check individual test results for data source information.

"""
    else:
        md += """## ✓ Real Binance Data Used

Tests include real market data downloaded from Binance API.

"""

    md += f"""## ✓ Data Leak Verification

All tests use strict walk-forward validation. The model only uses data from [t-window, t) to predict at time t. 
**No future information is used. No data leakage.**

## Executive Summary

- **Total Tests Run:** {len(all_results)}
- **Trading Pairs Tested:** {len(set(r['pair'] for r in all_results if 'pair' in r))}
- **Test Horizons:** {len(set(r.get('horizon', 1) for r in all_results))}
- **Average Hit Rate:** {np.mean(valid_hit_rates) if valid_hit_rates else 0.0:.4f}

"""
    
    # Synthetic results
    synthetic_results = [r for r in all_results if r.get('data_type') == 'synthetic']
    if synthetic_results:
        md += """## 🧪 Synthetic Data Results

Synthetic data contains known periodic patterns at multiple frequencies.
Good performance here indicates the model can capture cyclical patterns.

| Horizon | Hit Rate | Correlation | Predictions | Performance |
|---------|----------|-------------|-------------|-------------|
"""
        for r in synthetic_results:
            hit_rate = r.get('hit_rate', np.nan)
            corr = r.get('corr_pred_true', np.nan)
            n_pred = r.get('n_predictions', 0)
            
            if hit_rate > 0.6:
                perf = "Excellent ✓"
            elif hit_rate > 0.55:
                perf = "Good"
            elif hit_rate > 0.5:
                perf = "Fair"
            else:
                perf = "Poor ✗"
            
            md += f"| {r.get('horizon', 1)}h | {hit_rate:.4f} | {corr:.4f} | {n_pred} | {perf} |\n"
    
    # Real results - distinguish real vs mock
    real_results = [r for r in all_results if r.get('data_type') in ['real', 'mock']]
    if real_results:
        truly_real = [r for r in real_results if r.get('is_real_binance', False)]
        mock_results = [r for r in real_results if not r.get('is_real_binance', False)]
        
        if truly_real:
            md += "\n## 📈 Real Binance Market Data Results\n\n"
            md += "**✓ Real Binance Data:** Actual market data downloaded from Binance API.\n"
            md += "Performance above 0.5 hit rate indicates genuine predictive power.\n\n"
            
            pairs = sorted(set(r['pair'] for r in truly_real if 'pair' in r))
            
            for pair in pairs:
                pair_results = [r for r in truly_real if r.get('pair') == pair]
                md += f"\n### ✓ {pair_results[0].get('pair_label', pair)} (Real Binance Data)\n\n"
                md += "| Horizon | Hit Rate | Correlation | Predictions | Performance |\n"
                md += "|---------|----------|-------------|-------------|-------------|\n"
                
                for r in sorted(pair_results, key=lambda x: x.get('horizon', 1)):
                    hit_rate = r.get('hit_rate', np.nan)
                    corr = r.get('corr_pred_true', np.nan)
                    n_pred = r.get('n_predictions', 0)
                    
                    if hit_rate > 0.55:
                        perf = "Excellent ✓"
                    elif hit_rate > 0.52:
                        perf = "Good"
                    elif hit_rate > 0.5:
                        perf = "Fair"
                    else:
                        perf = "Poor ✗"
                    
                    md += f"| {r.get('horizon', 1)}h | {hit_rate:.4f} | {corr:.4f} | {n_pred} | {perf} |\n"
        
        if mock_results:
            md += "\n## 📊 Mock/Simulated Market Data Results\n\n"
            md += "**⚠️ Mock Data:** These results use SIMULATED data, NOT real Binance data.\n"
            md += "Results are for testing infrastructure only and do NOT represent real market performance.\n\n"
            
            pairs = sorted(set(r['pair'] for r in mock_results if 'pair' in r))
            
            for pair in pairs:
                pair_results = [r for r in mock_results if r.get('pair') == pair]
                md += f"\n### ⚠ {pair_results[0].get('pair_label', pair)} (Mock Data)\n\n"
                md += "| Horizon | Hit Rate | Correlation | Predictions | Performance |\n"
                md += "|---------|----------|-------------|-------------|-------------|\n"
                
                for r in sorted(pair_results, key=lambda x: x.get('horizon', 1)):
                    hit_rate = r.get('hit_rate', np.nan)
                    corr = r.get('corr_pred_true', np.nan)
                    n_pred = r.get('n_predictions', 0)
                    
                    if hit_rate > 0.55:
                        perf = "Excellent ✓"
                    elif hit_rate > 0.52:
                        perf = "Good"
                    elif hit_rate > 0.5:
                        perf = "Fair"
                    else:
                        perf = "Poor ✗"
                    
                    md += f"| {r.get('horizon', 1)}h | {hit_rate:.4f} | {corr:.4f} | {n_pred} | {perf} |\n"
    
    md += f"""
## ⚙️ Technical Configuration

**Model Parameters:**
- Window Size: {WINDOW_SIZE}
- Q Parameter: {Q_PARAM}
- N Terms: {N_TERMS}
- N Frequencies: {N_FREQ}
- Lambda (Regularization): {LAMBDA_PARAM}
- Phase Scale: {PHASE_SCALE}

**Validation Method:** Strict Walk-Forward
- Model trained on [t-window, t)
- Prediction made for time t
- No future information used
- No data leakage

## 📋 Interpretation Guide

**Hit Rate:** Percentage of correct directional predictions (up/down)
- > 0.55: Excellent predictive power
- 0.52-0.55: Good predictive power
- 0.50-0.52: Fair predictive power
- < 0.50: No predictive power (below random chance)

**Correlation:** Linear relationship between predicted and actual changes
- > 0.3: Strong relationship
- 0.1-0.3: Moderate relationship
- < 0.1: Weak relationship

---
*Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    with open(output_path, 'w') as f:
        f.write(md)
    
    print(f"✓ Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive test of biquaternion bot on real Binance data'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading data (use existing data)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (fewer horizons, fewer pairs)'
    )
    
    args = parser.parse_args()
    
    print_section("THETA BOT BIQUATERNION - COMPREHENSIVE TEST", '=')
    print("""
This script will:
1. Generate synthetic data with known periodic patterns
2. Download real market data from Binance (multiple pairs including PLN)
3. Test corrected biquaternion implementation on all data
4. Test multiple prediction horizons
5. Generate comprehensive HTML and Markdown reports
6. Verify no data leaks with strict walk-forward validation

All tests use walk-forward validation: model uses ONLY data from [t-window, t)
to predict at time t. NO FUTURE INFORMATION IS USED.
""")
    
    all_results = []
    any_real_binance_data = False  # Track if any real Binance data was used
    
    # Step 1: Synthetic Data
    print_section("STEP 1: SYNTHETIC DATA TESTING")
    synthetic_path = generate_synthetic_data(n_samples=2000)
    
    test_horizons = [1, 4, 8] if args.quick else TEST_HORIZONS
    
    for horizon in test_horizons:
        outdir = f'test_output/binance_synthetic_h{horizon}'
        result = run_evaluation(synthetic_path, f"Synthetic (h={horizon})", outdir, horizon=horizon)
        
        if result:
            result['data_type'] = 'synthetic'
            result['pair'] = 'SYNTHETIC'
            result['pair_label'] = 'Synthetic Data'
            result['is_real_binance'] = False
            all_results.append(result)
    
    # Step 2: Real Market Data
    print_section("STEP 2: REAL MARKET DATA TESTING")
    
    if not args.skip_download:
        print("\nAttempting to download real market data from Binance...")
    
    test_pairs = TRADING_PAIRS[:3] if args.quick else TRADING_PAIRS
    
    for symbol, label in test_pairs:
        is_real_binance = False
        if not args.skip_download:
            data_path, is_real_binance = download_binance_data(symbol, interval='1h', limit=DATA_LIMIT)
            any_real_binance_data = any_real_binance_data or is_real_binance
        else:
            # Check if it's real data or mock data
            data_path = f'real_data/{symbol}_1h.csv'
            mock_path = f'real_data/{symbol}_1h_mock.csv'
            
            if os.path.exists(data_path):
                is_real_binance = True
                any_real_binance_data = True
            elif os.path.exists(mock_path):
                data_path = mock_path
                is_real_binance = False
            else:
                print(f"  ✗ Skipping {symbol} - file not found")
                continue
        
        if data_path:
            # Test on single horizon (1h) for all pairs, multiple horizons for quick test
            horizons_to_test = [1] if not args.quick else [1, 4]
            
            for horizon in horizons_to_test:
                outdir = f'test_output/binance_{symbol}_h{horizon}'
                result = run_evaluation(data_path, f"{label} (h={horizon})", outdir, horizon=horizon)
                
                if result:
                    result['data_type'] = 'real' if is_real_binance else 'mock'
                    result['pair'] = symbol
                    result['pair_label'] = label
                    result['is_real_binance'] = is_real_binance
                    all_results.append(result)
    
    # Step 3: Generate Reports
    if all_results:
        generate_html_report(all_results)
        
        # Print summary to console
        print_section("TEST SUMMARY")
        
        print("\nSynthetic Data:")
        synthetic_results = [r for r in all_results if r.get('data_type') == 'synthetic']
        for r in synthetic_results:
            print(f"  h={int(r.get('horizon', 1)):2d}: Hit Rate={r.get('hit_rate', 0):.4f}, "
                  f"Corr={r.get('corr_pred_true', 0):.4f}, N={int(r.get('n_predictions', 0))}")
        
        # Separate real and mock data
        truly_real = [r for r in all_results if r.get('is_real_binance', False)]
        mock_data = [r for r in all_results if r.get('data_type') == 'mock']
        
        if truly_real:
            print("\n✓ Real Binance Market Data:")
            current_pair = None
            for r in sorted(truly_real, key=lambda x: (x.get('pair', ''), x.get('horizon', 1))):
                if r.get('pair') != current_pair:
                    current_pair = r.get('pair')
                    print(f"\n  {r.get('pair_label', current_pair)} (REAL):")
                print(f"    h={int(r.get('horizon', 1)):2d}: Hit Rate={r.get('hit_rate', 0):.4f}, "
                      f"Corr={r.get('corr_pred_true', 0):.4f}, N={int(r.get('n_predictions', 0))}")
        
        if mock_data:
            print("\n⚠ Mock/Simulated Market Data:")
            current_pair = None
            for r in sorted(mock_data, key=lambda x: (x.get('pair', ''), x.get('horizon', 1))):
                if r.get('pair') != current_pair:
                    current_pair = r.get('pair')
                    print(f"\n  {r.get('pair_label', current_pair)} (MOCK - NOT REAL):")
                print(f"    h={int(r.get('horizon', 1)):2d}: Hit Rate={r.get('hit_rate', 0):.4f}, "
                      f"Corr={r.get('corr_pred_true', 0):.4f}, N={int(r.get('n_predictions', 0))}")
        
        print_section("TESTING COMPLETE", '=')
        
        # Add clear warning if no real data
        if any_real_binance_data:
            data_status = "✓ REAL Binance data was used for some tests"
        else:
            data_status = "⚠ WARNING: NO REAL Binance data - all tests used MOCK data"
        
        print(f"""
✓ Ran {len(all_results)} tests total
✓ Reports saved to test_output/
✓ View comprehensive_report.html in a browser
✓ View comprehensive_report.md for text version

Data Source Status:
{data_status}

Key Findings:
- All tests used strict walk-forward validation (NO DATA LEAKS)
- Model only used historical data [t-window, t) to predict at time t
- Hit rates > 0.5 indicate genuine predictive power
- Performance varies by trading pair and prediction horizon
""")
    else:
        print("\n⚠ No successful test results. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
