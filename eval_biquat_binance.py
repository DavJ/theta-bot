#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_biquat_binance.py
---------------------
Properly evaluate the biquaternion model on Binance data.

This script:
1. Fetches real data from Binance (or uses local CSV)
2. Trains the biquaternion model using theta_eval_biquat_corrected.py
3. Generates predictions using the trained model
4. Evaluates performance with proper metrics

Usage:
    python eval_biquat_binance.py --symbol BTCUSDT --interval 1h --limit 1000
    python eval_biquat_binance.py --csv real_data/BTCUSDT_1h.csv
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime


def fetch_or_load_data(symbol=None, interval='1h', limit=1000, csv_path=None):
    """
    Fetch data from Binance or load from CSV.
    
    Returns:
        DataFrame with timestamp, open, high, low, close, volume
    """
    if csv_path and os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    if symbol:
        print(f"Fetching {symbol} data from Binance...")
        
        # Use the download script
        output_dir = 'real_data'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{symbol}_{interval}.csv")
        
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
                print(f"✓ Downloaded {len(df)} samples")
                return df
            else:
                print(f"✗ Failed to download {symbol}")
                print(result.stderr)
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    return None


def run_biquat_evaluation(csv_path, horizon=1, window=256, output_dir='eval_output'):
    """
    Run the biquaternion model evaluation.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nRunning biquaternion model evaluation...")
    print(f"  Horizon: {horizon}h")
    print(f"  Window: {window}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable,
        'theta_bot_averaging/theta_eval_biquat_corrected.py',
        '--csv', csv_path,
        '--price-col', 'close',
        '--horizon', str(horizon),
        '--window', str(window),
        '--q', '0.6',
        '--n-terms', '16',
        '--n-freq', '6',
        '--lambda', '0.5',
        '--phase-scale', '1.0',
        '--outdir', output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"✗ Evaluation failed")
            print(result.stderr)
            return None
        
        # Load summary
        summary_path = os.path.join(output_dir, 'summary.csv')
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            metrics = summary.iloc[0].to_dict()
            
            hit_rate = metrics.get('hit_rate', np.nan)
            correlation = metrics.get('corr_pred_true', np.nan)
            n_pred = metrics.get('n_predictions', 0)
            
            print(f"✓ Model Results:")
            print(f"  Hit Rate: {hit_rate:.4f}")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  Predictions: {n_pred}")
            
            return metrics
        else:
            print(f"✗ Summary file not found")
            return None
            
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return None


def simulate_trading(df, predictions_col='prediction', fee_rate=0.001, start_capital=1000.0):
    """
    Simulate trading based on model predictions.
    
    Args:
        df: DataFrame with 'close' prices and predictions
        predictions_col: Column name with predictions
        fee_rate: Taker fee rate (0.001 = 0.1%)
        start_capital: Starting capital in USDT
    
    Returns:
        Dictionary with trading metrics
    """
    if predictions_col not in df.columns:
        print(f"Warning: {predictions_col} column not found")
        return None
    
    capital = start_capital
    position = 0
    trades = []
    
    for i in range(1, len(df)):
        if pd.isna(df.iloc[i][predictions_col]):
            continue
        
        pred = df.iloc[i][predictions_col]
        current_price = df.iloc[i]['close']
        
        # Go long if prediction is positive
        if pred > 0 and position == 0:
            # Buy
            shares = capital / (current_price * (1 + fee_rate))
            position = shares
            capital = 0
            trades.append(('BUY', current_price, shares))
        elif pred <= 0 and position > 0:
            # Sell
            capital = position * current_price * (1 - fee_rate)
            trades.append(('SELL', current_price, position))
            position = 0
    
    # Close any open position
    if position > 0:
        final_price = df.iloc[-1]['close']
        capital = position * final_price * (1 - fee_rate)
    
    total_pnl = capital - start_capital
    
    # Compute monthly PnL
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


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate biquaternion model on Binance data'
    )
    parser.add_argument('--symbol', type=str, help='Binance symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h', help='Interval (default: 1h)')
    parser.add_argument('--limit', type=int, default=1000, help='Number of klines (default: 1000)')
    parser.add_argument('--csv', type=str, help='Path to CSV file (alternative to fetching)')
    parser.add_argument('--horizon', type=int, default=1, help='Prediction horizon in hours (default: 1)')
    parser.add_argument('--window', type=int, default=256, help='Training window size (default: 256)')
    parser.add_argument('--start-capital', type=float, default=1000.0, help='Starting capital (default: 1000)')
    parser.add_argument('--fee', type=float, default=0.001, help='Taker fee rate (default: 0.001)')
    parser.add_argument('--output', type=str, default='eval_output', help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("BIQUATERNION MODEL EVALUATION ON BINANCE DATA")
    print("="*70)
    print()
    
    # Step 1: Get data
    df = fetch_or_load_data(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        csv_path=args.csv
    )
    
    if df is None or len(df) < args.window + args.horizon + 10:
        print("Error: Insufficient data")
        return 1
    
    # Save to temporary CSV if needed
    temp_csv = None
    if args.csv:
        temp_csv = args.csv
    else:
        temp_csv = os.path.join(args.output, 'temp_data.csv')
        df.to_csv(temp_csv, index=False)
    
    # Step 2: Run biquaternion model
    metrics = run_biquat_evaluation(
        csv_path=temp_csv,
        horizon=args.horizon,
        window=args.window,
        output_dir=args.output
    )
    
    if metrics is None:
        print("\nError: Evaluation failed")
        return 1
    
    # Step 3: Load predictions and simulate trading
    print("\nSimulating trading strategy...")
    
    # Try to load predictions from output
    pred_file = os.path.join(args.output, 'predictions.csv')
    if os.path.exists(pred_file):
        pred_df = pd.read_csv(pred_file)
        
        # Merge with original data
        if 'timestamp' in df.columns and 'timestamp' in pred_df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
            df = df.merge(pred_df[['timestamp', 'prediction']], on='timestamp', how='left')
        
        # Simulate trading with no fees
        trading_no_fee = simulate_trading(df, fee_rate=0.0, start_capital=args.start_capital)
        
        # Simulate trading with fees
        trading_with_fee = simulate_trading(df, fee_rate=args.fee, start_capital=args.start_capital)
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print()
        print("Model Performance:")
        print(f"  Hit Rate: {metrics.get('hit_rate', 0):.4f}")
        print(f"  Correlation: {metrics.get('corr_pred_true', 0):.4f}")
        print(f"  Predictions: {int(metrics.get('n_predictions', 0))}")
        print()
        
        if trading_no_fee:
            print("Trading Performance (No Fees):")
            print(f"  Total PnL: ${trading_no_fee['total_pnl_usdt']:.2f}")
            print(f"  End Capital: ${trading_no_fee['end_capital_usdt']:.2f}")
            print(f"  Monthly PnL: ${trading_no_fee['avg_monthly_pnl_usdt']:.2f}")
            print(f"  Trades: {trading_no_fee['num_trades']}")
            print()
        
        if trading_with_fee:
            print(f"Trading Performance (With {args.fee*100:.2f}% Fees):")
            print(f"  Total PnL: ${trading_with_fee['total_pnl_usdt']:.2f}")
            print(f"  End Capital: ${trading_with_fee['end_capital_usdt']:.2f}")
            print(f"  Monthly PnL: ${trading_with_fee['avg_monthly_pnl_usdt']:.2f}")
            print(f"  Trades: {trading_with_fee['num_trades']}")
        
        print("\n" + "="*70)
        
    else:
        print("Warning: Predictions file not found, skipping trading simulation")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
