#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_market_data.py
-----------------------
Download real market data from Binance or use provided CSV files.

This script prepares real market data for testing the theta bot model
before production deployment.

Usage:
    python download_market_data.py --symbol BTCUSDT --interval 1h --limit 2000
    python download_market_data.py --csv path/to/BTCUSDT_1h.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def download_binance_data(symbol='BTCUSDT', interval='1h', limit=2000):
    """
    Download historical data from Binance API.
    
    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
    interval : str
        Timeframe interval (e.g., '1h', '4h', '1d')
    limit : int
        Number of candles to fetch
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with OHLCV data
    """
    try:
        import requests
    except ImportError:
        print("Error: 'requests' package not found. Install with: pip install requests")
        sys.exit(1)
    
    print(f"Downloading {symbol} {interval} data from Binance...")
    
    url = "https://api.binance.com/api/v3/klines"
    
    # Binance has a limit of 1000 per request
    all_data = []
    remaining = limit
    end_time = None
    
    while remaining > 0:
        batch_size = min(1000, remaining)
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': batch_size
        }
        
        if end_time:
            params['endTime'] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data = data + all_data
            remaining -= len(data)
            
            # Set end_time to the timestamp of the earliest candle - 1ms
            end_time = data[0][0] - 1
            
            print(f"  Downloaded {len(data)} candles, {remaining} remaining...")
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            if all_data:
                print(f"Continuing with {len(all_data)} candles already downloaded...")
                break
            else:
                sys.exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Keep only relevant columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"Downloaded {len(df)} candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
    
    return df


def load_csv_data(csv_path):
    """
    Load market data from CSV file.
    
    Expected columns: timestamp, open, high, low, close, volume
    Or at minimum: timestamp, close
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame with market data
    """
    print(f"Loading data from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    if 'close' not in df.columns:
        print("Error: CSV must contain 'close' column")
        sys.exit(1)
    
    # Try to parse timestamp if present
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            print("Warning: Could not parse timestamp column")
    elif 'date' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['date'])
        except:
            print("Warning: Could not parse date column")
    else:
        # Create synthetic timestamps
        print("Warning: No timestamp column found, creating synthetic timestamps")
        df['timestamp'] = pd.date_range('2023-01-01', periods=len(df), freq='h')
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in close price
    initial_len = len(df)
    df = df.dropna(subset=['close'])
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} rows with NaN close prices")
    
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")
    
    return df


def validate_data(df):
    """
    Validate that data is suitable for testing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data
        
    Returns
    -------
    valid : bool
        Whether data is valid
    """
    print("\nValidating data...")
    
    valid = True
    
    # Check minimum length
    if len(df) < 1000:
        print(f"  ⚠ Warning: Only {len(df)} samples (recommended: ≥1000)")
        valid = False
    else:
        print(f"  ✓ Sample count: {len(df)}")
    
    # Check for missing values
    missing = df['close'].isna().sum()
    if missing > 0:
        print(f"  ⚠ Warning: {missing} missing close prices")
        valid = False
    else:
        print(f"  ✓ No missing values")
    
    # Check for constant values
    if df['close'].std() == 0:
        print(f"  ✗ Error: Close prices are constant")
        valid = False
    else:
        print(f"  ✓ Price variation detected (std={df['close'].std():.2f})")
    
    # Check for reasonable price range
    price_range = df['close'].max() / df['close'].min()
    if price_range > 100:
        print(f"  ⚠ Warning: Very large price range ({price_range:.1f}x)")
    else:
        print(f"  ✓ Price range: {price_range:.2f}x")
    
    return valid


def main():
    parser = argparse.ArgumentParser(
        description='Download or prepare real market data for theta bot testing'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading pair symbol (default: BTCUSDT)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        help='Timeframe interval (default: 1h)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=2000,
        help='Number of candles to download (default: 2000)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to existing CSV file to use instead of downloading'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV filename (default: auto-generated)'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='real_data',
        help='Output directory (default: real_data)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load or download data
    if args.csv:
        df = load_csv_data(args.csv)
        symbol = os.path.splitext(os.path.basename(args.csv))[0]
    else:
        df = download_binance_data(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit
        )
        symbol = f"{args.symbol}_{args.interval}"
    
    # Validate data
    if not validate_data(df):
        print("\n⚠ Data validation warnings detected. Continue anyway? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            print("Aborted.")
            sys.exit(1)
    
    # Save to output
    if args.output:
        output_path = os.path.join(args.outdir, args.output)
    else:
        output_path = os.path.join(args.outdir, f"{symbol}.csv")
    
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")
    
    # Print next steps
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print(f"1. Run validation tests:")
    print(f"   python validate_real_data.py --csv {output_path}")
    print(f"\n2. Run predictions with different horizons:")
    print(f"   python theta_predictor.py --csv {output_path} --window 512 --outdir test_output")
    print(f"\n3. Run control tests (permutation and noise):")
    print(f"   python theta_horizon_scan_updated.py --csv {output_path} --test-controls --outdir test_output")
    print(f"\n4. Optimize hyperparameters:")
    print(f"   python optimize_hyperparameters.py --csv {output_path} --outdir test_output")
    print("="*70)


if __name__ == '__main__':
    main()
