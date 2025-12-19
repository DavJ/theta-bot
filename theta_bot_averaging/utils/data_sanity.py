"""
Data sanity checks for price datasets.

Verifies that data is realistic and not synthetic or improperly scaled.
"""

from __future__ import annotations

from typing import Dict
import pandas as pd
import numpy as np


def check_price_sanity(df: pd.DataFrame, symbol: str = "BTCUSDT") -> Dict:
    """
    Check if price data appears realistic for the given symbol and time period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and 'close' column
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT')
    
    Returns
    -------
    dict
        Dictionary with:
        - min_close: Minimum close price
        - max_close: Maximum close price
        - mean_close: Mean close price
        - start_timestamp: First timestamp
        - end_timestamp: Last timestamp
        - num_rows: Number of rows
        - appears_synthetic: Boolean flag
        - warning_message: Warning message if data appears synthetic
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must have 'close' column")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    min_close = float(df['close'].min())
    max_close = float(df['close'].max())
    mean_close = float(df['close'].mean())
    start_ts = df.index[0]
    end_ts = df.index[-1]
    num_rows = len(df)
    
    # Check if data appears synthetic or unrealistic
    appears_synthetic = False
    warning_message = None
    
    # For BTCUSDT in 2024, realistic range is roughly $15,000 - $110,000
    # Data outside this range (or with extreme scaling) is likely synthetic
    if symbol.upper() == "BTCUSDT":
        year_start = start_ts.year
        year_end = end_ts.year
        
        # 2024 BTC price ranges (approximate)
        if year_start >= 2024 or year_end >= 2024:
            # BTC was around $40k-$70k+ in 2024
            if min_close < 10000 or max_close > 120000:
                appears_synthetic = True
                warning_message = (
                    f"BTCUSDT price range ${min_close:.2f} - ${max_close:.2f} "
                    f"appears unrealistic for {year_start}-{year_end}. "
                    f"Expected range: ~$15,000 - $110,000. "
                    f"Data may be synthetic or improperly scaled."
                )
            # Check for unusual scaling patterns
            elif min_close < 15000 and max_close > 100000:
                # Wide range might indicate concatenated synthetic data
                appears_synthetic = True
                warning_message = (
                    f"BTCUSDT price range ${min_close:.2f} - ${max_close:.2f} "
                    f"spans an unusually wide range for {year_start}-{year_end}. "
                    f"This may indicate synthetic or concatenated data."
                )
    
    return {
        'min_close': min_close,
        'max_close': max_close,
        'mean_close': mean_close,
        'start_timestamp': start_ts,
        'end_timestamp': end_ts,
        'num_rows': num_rows,
        'appears_synthetic': appears_synthetic,
        'appears_unrealistic': appears_synthetic,  # Alias for clarity
        'warning_message': warning_message,
    }


def log_data_sanity(df: pd.DataFrame, symbol: str = "BTCUSDT") -> Dict:
    """
    Check and log data sanity information.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and 'close' column
    symbol : str
        Trading pair symbol
    
    Returns
    -------
    dict
        Sanity check results (same as check_price_sanity)
    """
    stats = check_price_sanity(df, symbol)
    
    print("\n" + "=" * 70)
    print("DATA SANITY CHECK")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Date Range: {stats['start_timestamp']} to {stats['end_timestamp']}")
    print(f"Number of Rows: {stats['num_rows']:,}")
    print(f"Price Range: ${stats['min_close']:.2f} to ${stats['max_close']:.2f}")
    print(f"Mean Price: ${stats['mean_close']:.2f}")
    
    if stats['appears_synthetic']:
        print("\n⚠️  WARNING: DATA APPEARS SYNTHETIC OR UNREALISTIC")
        print(f"    {stats['warning_message']}")
    else:
        print("\n✓ Data appears realistic for the given time period")
    
    print("=" * 70 + "\n")
    
    return stats
