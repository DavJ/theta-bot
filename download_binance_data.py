#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_binance_data.py
------------------------
Utility script to fetch Binance klines for given symbols/intervals/days
and save CSVs to real_data/ directory.

This is a focused utility for downloading historical kline data from Binance
public API and storing it in CSV format for evaluation and testing.

Usage:
    python3 download_binance_data.py --symbols BTCUSDT ETHUSDT --interval 1h --days 30
    python3 download_binance_data.py --symbols BNBUSDT --interval 15m --days 7
    python3 download_binance_data.py --help
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def fetch_binance_klines(symbol, interval='1h', start_time=None, end_time=None, limit=1000):
    """
    Fetch klines (candlestick data) from Binance public API.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1h', '15m', '1d')
        start_time: Start timestamp in milliseconds (optional)
        end_time: End timestamp in milliseconds (optional)
        limit: Number of klines to fetch per request (max 1000)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        import requests
    except ImportError:
        print("Error: 'requests' package required. Install with: pip install requests")
        sys.exit(1)
    
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return None
        
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
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}", file=sys.stderr)
        return None


def download_historical_data(symbol, interval='1h', days=30, output_dir='real_data'):
    """
    Download historical kline data for specified number of days.
    
    Args:
        symbol: Trading pair symbol
        interval: Kline interval
        days: Number of days of historical data to fetch
        output_dir: Directory to save CSV files
    
    Returns:
        Path to saved CSV file or None if failed
    """
    print(f"Downloading {symbol} {interval} data for last {days} days...")
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    end_ms = int(end_time.timestamp() * 1000)
    start_ms = int(start_time.timestamp() * 1000)
    
    all_data = []
    current_start = start_ms
    
    # Binance limits to 1000 klines per request
    # We need to paginate if requesting more data
    while current_start < end_ms:
        df = fetch_binance_klines(
            symbol=symbol,
            interval=interval,
            start_time=current_start,
            end_time=end_ms,
            limit=1000
        )
        
        if df is None or len(df) == 0:
            break
        
        all_data.append(df)
        
        # Move start time to after the last fetched candle
        last_timestamp = df['timestamp'].iloc[-1]
        current_start = int(last_timestamp.timestamp() * 1000) + 1
        
        print(f"  Fetched {len(df)} klines (total: {sum(len(d) for d in all_data)})")
        
        # If we got fewer than requested, we've reached the end
        if len(df) < 1000:
            break
    
    if not all_data:
        print(f"No data fetched for {symbol}")
        return None
    
    # Combine all data
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates (if any)
    full_df = full_df.drop_duplicates(subset=['timestamp'])
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{symbol}_{interval}.csv")
    full_df.to_csv(output_file, index=False)
    
    print(f"✓ Saved {len(full_df)} klines to {output_file}")
    print(f"  Date range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")
    print(f"  Price range: {full_df['close'].min():.2f} - {full_df['close'].max():.2f}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Download Binance historical kline data to CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BTC and ETH hourly data for last 30 days
  python3 download_binance_data.py --symbols BTCUSDT ETHUSDT --interval 1h --days 30
  
  # Download BNB 15-minute data for last week
  python3 download_binance_data.py --symbols BNBUSDT --interval 15m --days 7
  
  # Download multiple pairs with daily candles
  python3 download_binance_data.py --symbols BTCUSDT ETHUSDT BNBUSDT --interval 1d --days 365
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        required=True,
        help='Trading pair symbols (e.g., BTCUSDT ETHUSDT)'
    )
    parser.add_argument(
        '--interval',
        default='1h',
        help='Kline interval (e.g., 1m, 5m, 15m, 1h, 4h, 1d). Default: 1h'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days of historical data to fetch. Default: 30'
    )
    parser.add_argument(
        '--output-dir',
        default='real_data',
        help='Output directory for CSV files. Default: real_data'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Binance Data Downloader")
    print("="*70)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Interval: {args.interval}")
    print(f"Days: {args.days}")
    print(f"Output Directory: {args.output_dir}")
    print("="*70)
    print()
    
    success_count = 0
    failed_symbols = []
    
    for symbol in args.symbols:
        try:
            result = download_historical_data(
                symbol=symbol,
                interval=args.interval,
                days=args.days,
                output_dir=args.output_dir
            )
            if result:
                success_count += 1
            else:
                failed_symbols.append(symbol)
        except Exception as e:
            print(f"✗ Failed to download {symbol}: {e}")
            failed_symbols.append(symbol)
        print()
    
    # Summary
    print("="*70)
    print("Download Summary")
    print("="*70)
    print(f"Successfully downloaded: {success_count}/{len(args.symbols)} symbols")
    if failed_symbols:
        print(f"Failed symbols: {', '.join(failed_symbols)}")
    print("="*70)
    
    return 0 if success_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
