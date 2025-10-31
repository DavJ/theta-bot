#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_metrics.py
---------------
Evaluation script for computing performance metrics on Binance data.

Computes:
- earnings (total PnL in USDT)
- end_capital_usdt (final capital after trades)
- avg_monthly_pnl_usdt (average monthly PnL)
- corr_pred_true (Pearson correlation between predictions and true returns)
- hit_rate (fraction of correct direction predictions, zeros count as misses)

Supports:
- Multiple trading pairs
- Fee modes: no_fees and taker_fee (0.001 = 0.10% per side)
- Fetches data from Binance public REST API

Usage:
    python3 tools/eval_metrics.py --repo-root . --start-capital 1000 --taker-fee 0.001
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path


def fetch_binance_klines(symbol, interval='1h', limit=1000):
    """
    Fetch klines (candlestick data) from Binance public API.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (default '1h')
        limit: Number of klines to fetch (max 1000 per request)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse klines data
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to appropriate types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Keep only essential columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {symbol}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error parsing {symbol} data: {e}", file=sys.stderr)
        return None


def find_dataset_files(repo_root):
    """
    Find CSV dataset files in the repository.
    
    Returns:
        List of tuples: (file_path, dataset_name, dataset_type, pair_name)
    """
    datasets = []
    repo_path = Path(repo_root)
    seen_files = set()
    
    # Check common data directories
    data_dirs = [
        repo_path / 'real_data',
        repo_path / 'history',
        repo_path / 'test_output',
        repo_path / 'test_data',
        repo_path
    ]
    
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        
        for csv_file in data_dir.glob('*.csv'):  # Only direct children, not recursive
            # Avoid processing the same file twice
            if str(csv_file) in seen_files:
                continue
            seen_files.add(str(csv_file))
            
            # Skip files that are clearly not price data
            if any(skip in csv_file.name.lower() for skip in ['summary', 'report', 'eval', 'metrics']):
                continue
            
            # Quick check: does it have a 'close' column?
            try:
                df_test = pd.read_csv(csv_file, nrows=1)
                if 'close' not in df_test.columns:
                    continue
            except:
                continue
            
            # Determine dataset type and extract pair name
            file_name = csv_file.name
            dataset_type = 'real'
            
            if 'mock' in file_name.lower() or 'synthetic' in file_name.lower():
                dataset_type = 'synthetic'
            
            # Extract pair name (e.g., BTCUSDT from BTCUSDT_1h.csv)
            base_name = file_name.replace('.csv', '').split('_')[0].upper()
            
            # Check if it looks like a trading pair
            if len(base_name) >= 6 and any(quote in base_name for quote in ['USDT', 'USD', 'BTC', 'ETH', 'PLN']):
                pair_name = base_name
            else:
                # Generic name for synthetic or other data
                pair_name = 'SYNTH-PAIR'
            
            datasets.append((str(csv_file), file_name, dataset_type, pair_name))
    
    return datasets


def compute_returns(prices):
    """Compute simple returns from price series."""
    return np.diff(prices) / prices[:-1]


def compute_correlation(pred_returns, true_returns):
    """
    Compute Pearson correlation between predicted and true returns.
    
    Args:
        pred_returns: Predicted returns
        true_returns: True (realized) returns
    
    Returns:
        Pearson correlation coefficient or NaN if cannot compute
    """
    # Remove NaN values
    mask = ~(np.isnan(pred_returns) | np.isnan(true_returns))
    if mask.sum() < 2:
        return np.nan
    
    pred_clean = pred_returns[mask]
    true_clean = true_returns[mask]
    
    # Check for zero variance
    if np.std(pred_clean) == 0 or np.std(true_clean) == 0:
        return np.nan
    
    return np.corrcoef(pred_clean, true_clean)[0, 1]


def compute_hit_rate(pred_returns, true_returns):
    """
    Compute hit rate (fraction of correct direction predictions).
    Zeros in true_returns are counted as misses.
    
    Args:
        pred_returns: Predicted returns
        true_returns: True (realized) returns
    
    Returns:
        Hit rate (0.0 to 1.0) or NaN if cannot compute
    """
    # Remove NaN values
    mask = ~(np.isnan(pred_returns) | np.isnan(true_returns))
    if mask.sum() == 0:
        return np.nan
    
    pred_clean = pred_returns[mask]
    true_clean = true_returns[mask]
    
    # Zeros in true_returns count as misses (no clear direction)
    # So we only count as hit when signs match AND true_return != 0
    hits = 0
    total = len(true_clean)
    
    for pred, true_val in zip(pred_clean, true_clean):
        if true_val == 0:
            # Zero true return counts as a miss
            continue
        elif np.sign(pred) == np.sign(true_val):
            hits += 1
    
    if total == 0:
        return np.nan
    
    return hits / total


def simulate_trading(df, fee_rate=0.0, start_capital=1000.0):
    """
    Simulate trading with a simple strategy based on predicted returns.
    
    Args:
        df: DataFrame with 'close' prices and optionally 'predicted_return'
        fee_rate: Taker fee rate (e.g., 0.001 for 0.1%)
        start_capital: Starting capital in USDT
    
    Returns:
        Dictionary with trading metrics
    """
    if 'predicted_return' not in df.columns:
        # Generate dummy predictions based on momentum if no predictions available
        df = df.copy()
        returns = df['close'].pct_change()
        df['predicted_return'] = returns.shift(1)  # Simple momentum
    
    capital = start_capital
    position = 0  # 0 = no position, 1 = long, -1 = short (if supported)
    trades = []
    
    for i in range(1, len(df)):
        if pd.isna(df.iloc[i]['predicted_return']):
            continue
        
        pred_return = df.iloc[i]['predicted_return']
        current_price = df.iloc[i]['close']
        
        # Simple strategy: go long if predicted return > 0
        if pred_return > 0 and position == 0:
            # Buy (long)
            shares = capital / (current_price * (1 + fee_rate))
            position = shares
            capital = 0
            trades.append({
                'timestamp': df.iloc[i]['timestamp'],
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'fee': shares * current_price * fee_rate
            })
        elif pred_return <= 0 and position > 0:
            # Sell
            capital = position * current_price * (1 - fee_rate)
            trades.append({
                'timestamp': df.iloc[i]['timestamp'],
                'action': 'SELL',
                'price': current_price,
                'shares': position,
                'fee': position * current_price * fee_rate
            })
            position = 0
    
    # Close any open position at the end
    if position > 0:
        final_price = df.iloc[-1]['close']
        capital = position * final_price * (1 - fee_rate)
        position = 0
    
    total_pnl = capital - start_capital
    
    # Compute average monthly PnL
    if len(df) > 0 and 'timestamp' in df.columns:
        time_span_days = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).total_seconds() / 86400
        months = time_span_days / 30.0
        avg_monthly_pnl = total_pnl / months if months > 0 else 0
    else:
        avg_monthly_pnl = 0
    
    return {
        'total_pnl_usdt': total_pnl,
        'end_capital_usdt': capital,
        'avg_monthly_pnl_usdt': avg_monthly_pnl,
        'num_trades': len(trades)
    }


def evaluate_dataset(csv_path, fee_mode='no_fees', taker_fee=0.001, start_capital=1000.0):
    """
    Evaluate a single dataset file.
    
    Args:
        csv_path: Path to CSV file with price data
        fee_mode: 'no_fees' or 'taker_fee'
        taker_fee: Taker fee rate (e.g., 0.001 for 0.1%)
        start_capital: Starting capital in USDT
    
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure we have required columns
        if 'close' not in df.columns:
            return None
        
        # Parse timestamp if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            # Generate dummy timestamps
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Compute true returns
        true_returns = compute_returns(df['close'].values)
        
        # Check if we have predictions
        pred_col = None
        for col in ['predicted_return', 'pred_return', 'prediction']:
            if col in df.columns:
                pred_col = col
                break
        
        if pred_col:
            pred_returns = df[pred_col].values[1:]  # Align with true_returns
        else:
            # Use simple momentum as fallback
            momentum = df['close'].pct_change().shift(1).values[1:]
            pred_returns = momentum
            df['predicted_return'] = df['close'].pct_change().shift(1)
        
        # Ensure same length
        min_len = min(len(pred_returns), len(true_returns))
        pred_returns = pred_returns[:min_len]
        true_returns = true_returns[:min_len]
        
        # Compute metrics
        corr = compute_correlation(pred_returns, true_returns)
        hit_rate = compute_hit_rate(pred_returns, true_returns)
        
        # Determine fee rate
        fee_rate = taker_fee if fee_mode == 'taker_fee' else 0.0
        
        # Simulate trading
        trading_metrics = simulate_trading(df, fee_rate=fee_rate, start_capital=start_capital)
        
        return {
            'corr_pred_true': corr,
            'hit_rate': hit_rate,
            **trading_metrics
        }
        
    except Exception as e:
        print(f"Error evaluating {csv_path}: {e}", file=sys.stderr)
        return None


def evaluate_binance_pair(pair, interval='1h', limit=1000, fee_mode='no_fees', 
                          taker_fee=0.001, start_capital=1000.0):
    """
    Fetch data from Binance and evaluate.
    
    Args:
        pair: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval
        limit: Number of klines to fetch
        fee_mode: 'no_fees' or 'taker_fee'
        taker_fee: Taker fee rate
        start_capital: Starting capital in USDT
    
    Returns:
        Dictionary with evaluation metrics
    """
    df = fetch_binance_klines(pair, interval=interval, limit=limit)
    if df is None or len(df) < 2:
        return None
    
    # Add simple momentum predictions
    df['predicted_return'] = df['close'].pct_change().shift(1)
    
    # Compute true returns
    true_returns = compute_returns(df['close'].values)
    pred_returns = df['predicted_return'].values[1:]
    
    # Ensure same length
    min_len = min(len(pred_returns), len(true_returns))
    pred_returns = pred_returns[:min_len]
    true_returns = true_returns[:min_len]
    
    # Compute metrics
    corr = compute_correlation(pred_returns, true_returns)
    hit_rate = compute_hit_rate(pred_returns, true_returns)
    
    # Determine fee rate
    fee_rate = taker_fee if fee_mode == 'taker_fee' else 0.0
    
    # Simulate trading
    trading_metrics = simulate_trading(df, fee_rate=fee_rate, start_capital=start_capital)
    
    return {
        'corr_pred_true': corr,
        'hit_rate': hit_rate,
        **trading_metrics
    }


def format_metrics_table(results):
    """
    Format results as a Markdown table.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Markdown-formatted table string
    """
    # Table header
    header = "| dataset | dataset_type | pair | fee_mode | total_pnl_usdt | end_capital_usdt | avg_monthly_pnl_usdt | corr_pred_true | hit_rate |"
    separator = "|---------|--------------|------|----------|----------------|------------------|----------------------|----------------|----------|"
    
    lines = [header, separator]
    
    for result in results:
        line = (
            f"| {result['dataset']:40} | "
            f"{result['dataset_type']:12} | "
            f"{result['pair']:8} | "
            f"{result['fee_mode']:10} | "
            f"{result['total_pnl_usdt']:14.2f} | "
            f"{result['end_capital_usdt']:16.2f} | "
            f"{result['avg_monthly_pnl_usdt']:20.2f} | "
            f"{result['corr_pred_true']:14.4f} | "
            f"{result['hit_rate']:8.4f} |"
        )
        lines.append(line)
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Evaluate metrics on Binance data')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--start-capital', type=float, default=1000.0, help='Starting capital in USDT')
    parser.add_argument('--taker-fee', type=float, default=0.001, help='Taker fee rate (e.g., 0.001 for 0.1%%)')
    parser.add_argument('--pairs', nargs='+', default=['BTCUSDT', 'ETHUSDT'], help='Binance pairs to evaluate')
    parser.add_argument('--interval', default='1h', help='Kline interval')
    parser.add_argument('--limit', type=int, default=1000, help='Number of klines to fetch')
    parser.add_argument('--output-dir', default='test_output', help='Output directory for summary')
    
    args = parser.parse_args()
    
    print(f"Evaluation Script Started")
    print(f"Start Capital: ${args.start_capital:.2f}")
    print(f"Taker Fee: {args.taker_fee * 100:.3f}% per side")
    print()
    
    results = []
    
    # Evaluate local dataset files
    print("Searching for local dataset files...")
    datasets = find_dataset_files(args.repo_root)
    print(f"Found {len(datasets)} dataset files")
    print()
    
    for csv_path, dataset_name, dataset_type, pair in datasets:
        print(f"Evaluating {dataset_name}...")
        
        # Evaluate with no fees
        metrics_no_fee = evaluate_dataset(csv_path, fee_mode='no_fees', 
                                          taker_fee=args.taker_fee, 
                                          start_capital=args.start_capital)
        if metrics_no_fee:
            results.append({
                'dataset': dataset_name,
                'dataset_type': dataset_type,
                'pair': pair,
                'fee_mode': 'no_fees',
                **metrics_no_fee
            })
        
        # Evaluate with taker fees
        metrics_with_fee = evaluate_dataset(csv_path, fee_mode='taker_fee', 
                                           taker_fee=args.taker_fee, 
                                           start_capital=args.start_capital)
        if metrics_with_fee:
            results.append({
                'dataset': dataset_name,
                'dataset_type': dataset_type,
                'pair': pair,
                'fee_mode': 'taker_fee',
                **metrics_with_fee
            })
    
    # Evaluate Binance pairs
    print("Fetching data from Binance API...")
    for pair in args.pairs:
        print(f"Evaluating {pair}...")
        
        # Evaluate with no fees
        metrics_no_fee = evaluate_binance_pair(pair, interval=args.interval, 
                                               limit=args.limit, fee_mode='no_fees',
                                               taker_fee=args.taker_fee, 
                                               start_capital=args.start_capital)
        if metrics_no_fee:
            results.append({
                'dataset': f'{pair}_{args.interval}_binance',
                'dataset_type': 'binance_live',
                'pair': pair,
                'fee_mode': 'no_fees',
                **metrics_no_fee
            })
        
        # Evaluate with taker fees
        metrics_with_fee = evaluate_binance_pair(pair, interval=args.interval, 
                                                 limit=args.limit, fee_mode='taker_fee',
                                                 taker_fee=args.taker_fee, 
                                                 start_capital=args.start_capital)
        if metrics_with_fee:
            results.append({
                'dataset': f'{pair}_{args.interval}_binance',
                'dataset_type': 'binance_live',
                'pair': pair,
                'fee_mode': 'taker_fee',
                **metrics_with_fee
            })
    
    # Format and output results
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    
    if results:
        table = format_metrics_table(results)
        print(table)
        print()
        
        # Save to file
        output_dir = Path(args.repo_root) / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'eval_summary.md'
        
        with open(output_file, 'w') as f:
            f.write("# Evaluation Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Configuration:**\n")
            f.write(f"- Start Capital: ${args.start_capital:.2f}\n")
            f.write(f"- Taker Fee: {args.taker_fee * 100:.3f}% per side\n")
            f.write(f"- Interval: {args.interval}\n")
            f.write(f"- Limit: {args.limit} klines\n\n")
            f.write("## Results\n\n")
            f.write(table)
            f.write("\n\n")
            f.write("## Metric Definitions\n\n")
            f.write("- **total_pnl_usdt**: Total profit/loss in USDT\n")
            f.write("- **end_capital_usdt**: Final capital after trading\n")
            f.write("- **avg_monthly_pnl_usdt**: Average monthly profit/loss\n")
            f.write("- **corr_pred_true**: Pearson correlation between predicted and true returns\n")
            f.write("- **hit_rate**: Fraction of correct direction predictions (zeros count as misses)\n")
        
        print(f"Results saved to: {output_file}")
    else:
        print("No results to display")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
