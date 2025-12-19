"""
Data sanity checks for price datasets.

Verifies that data is realistic and not synthetic or improperly scaled.
"""

from __future__ import annotations

from typing import Dict
import pandas as pd
import numpy as np


def check_price_sanity(df: pd.DataFrame, symbol: str = "BTCUSDT", strict: bool = False) -> Dict:
    """
    Check if price data appears realistic for the given symbol and time period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and OHLCV columns
    symbol : str
        Trading pair symbol (e.g., 'BTCUSDT')
    strict : bool
        If True, raises ValueError when data fails sanity checks
    
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
        - is_realistic: Boolean flag (False if data fails any check)
        - appears_synthetic: Boolean flag (deprecated, use is_realistic)
        - appears_unrealistic: Boolean flag (True if data fails any check)
        - warning_message: Warning message if data appears unrealistic
        - failed_checks: List of failed check descriptions
    
    Raises
    ------
    ValueError
        If strict=True and data fails sanity checks
    """
    # Required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must have '{col}' column")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Basic statistics
    min_close = float(df['close'].min())
    max_close = float(df['close'].max())
    mean_close = float(df['close'].mean())
    start_ts = df.index[0]
    end_ts = df.index[-1]
    num_rows = len(df)
    
    # Run sanity checks
    failed_checks = []
    
    # Check 1: Row count > 1000
    if num_rows <= 1000:
        failed_checks.append(f"Insufficient data: {num_rows} rows (need > 1000)")
    
    # Check 2: No NaNs in OHLCV columns
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            failed_checks.append(f"Column '{col}' has {nan_count} NaN values")
    
    # Check 3: Prices must be positive
    if min_close <= 0 or df[required_cols[:-1]].min().min() <= 0:
        failed_checks.append("Prices contain zero or negative values")
    
    # Check 4: Timestamp monotonic increasing
    if not df.index.is_monotonic_increasing:
        failed_checks.append("Timestamps are not monotonically increasing")
    
    # Check 5: Realistic BTC price range (max/min < 10)
    price_range_ratio = max_close / min_close if min_close > 0 else float('inf')
    if price_range_ratio >= 10:
        failed_checks.append(
            f"Price range too wide: max/min = {price_range_ratio:.2f} >= 10 "
            f"(${min_close:.2f} to ${max_close:.2f})"
        )
    
    # Check 6: Symbol-specific checks
    if symbol.upper() == "BTCUSDT":
        year_start = start_ts.year
        year_end = end_ts.year
        
        # For 2024+ data, expect reasonable BTC price ranges
        if year_start >= 2024 or year_end >= 2024:
            # BTC was around $40k-$70k+ in 2024
            if min_close < 10000 or max_close > 120000:
                failed_checks.append(
                    f"BTCUSDT price range ${min_close:.2f} - ${max_close:.2f} "
                    f"appears unrealistic for {year_start}-{year_end}. "
                    f"Expected range: ~$15,000 - $110,000."
                )
            # Warn if min_close < 20000 for 2024+ data (optional check)
            elif min_close < 20000:
                # This is a warning, not a failure
                pass  # Could add to warnings list if we had one
    
    # Determine if data is realistic
    is_realistic = len(failed_checks) == 0
    appears_unrealistic = not is_realistic
    
    # Build warning message
    warning_message = None
    if not is_realistic:
        warning_message = "Dataset does not look real; failing validation. Reasons:\n" + "\n".join(f"  - {check}" for check in failed_checks)
    
    result = {
        'min_close': min_close,
        'max_close': max_close,
        'mean_close': mean_close,
        'start_timestamp': start_ts,
        'end_timestamp': end_ts,
        'num_rows': num_rows,
        'is_realistic': is_realistic,
        'appears_synthetic': appears_unrealistic,  # Deprecated, kept for backwards compat
        'appears_unrealistic': appears_unrealistic,
        'warning_message': warning_message,
        'failed_checks': failed_checks,
    }
    
    # Raise error if strict mode and data is unrealistic
    if strict and not is_realistic:
        raise ValueError(f"Dataset does not look real; aborting evaluation.\n{warning_message}")
    
    return result


def log_data_sanity(df: pd.DataFrame, symbol: str = "BTCUSDT", strict: bool = False) -> Dict:
    """
    Check and log data sanity information.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and OHLCV columns
    symbol : str
        Trading pair symbol
    strict : bool
        If True, raises ValueError when data fails sanity checks
    
    Returns
    -------
    dict
        Sanity check results (same as check_price_sanity)
    
    Raises
    ------
    ValueError
        If strict=True and data fails sanity checks
    """
    stats = check_price_sanity(df, symbol, strict=strict)
    
    print("\n" + "=" * 70)
    print("DATA SANITY CHECK")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Date Range: {stats['start_timestamp']} to {stats['end_timestamp']}")
    print(f"Number of Rows: {stats['num_rows']:,}")
    print(f"Price Range: ${stats['min_close']:.2f} to ${stats['max_close']:.2f}")
    print(f"Mean Price: ${stats['mean_close']:.2f}")
    
    if not stats['is_realistic']:
        print("\n⚠️  WARNING: DATA APPEARS SYNTHETIC OR UNREALISTIC")
        print(f"    Failed checks:")
        for check in stats['failed_checks']:
            print(f"      - {check}")
        if strict:
            print("\n    STRICT MODE: Evaluation will be aborted!")
    else:
        print("\n✓ Data appears realistic for the given time period")
        print("✓ All sanity checks passed")
    
    print("=" * 70 + "\n")
    
    return stats
