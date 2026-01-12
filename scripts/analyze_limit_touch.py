#!/usr/bin/env python3
"""
Limit Touch Analysis - Diagnostic Tool for Mean Reversion Thresholds

PURPOSE:
========
This script analyzes historical price data to determine the realistic probability
that LIMIT orders at various threshold distances will be touched/filled.
This helps answer: "What LIMIT threshold has a real chance of being filled?"

KEY INSIGHTS FOR LIMIT STRATEGIES:
===================================
A LIMIT order strategy can only be profitable when:
    threshold_bps > (fees + spread + slippage + min_profit)

Example:
- If fees = 0.1% (10 bps) per trade
- Round-trip fees = 0.2% (20 bps)
- Spread = 0.02% (2 bps)
- Slippage = 0.05% (5 bps)
- Minimum profit target = 0.1% (10 bps)
=> Minimum viable threshold ≈ 37 bps

However, larger thresholds reduce fill probability:
- Too small threshold → frequent fills but death by fees
- Too large threshold → no fills, no trades at all
- Zero trades is NOT a bug - it's market reality when threshold is too wide

IMPORTANT:
==========
This is a DIAGNOSTIC tool only. It does NOT:
- Modify the trading engine
- Add hysteresis parameters
- Change edge detection
- Alter Kalman filters
- Modify risk management
- Change intent generation

It simply measures historical price reversion to inform threshold selection.

USAGE:
======
    python scripts/analyze_limit_touch.py --csv data/ohlcv_1h/BTCUSDT.csv
    python scripts/analyze_limit_touch.py --csv data.csv --tail 1000 --out results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    """
    Load OHLCV CSV file.
    
    Expected columns: timestamp, open, high, low, close, volume
    
    Args:
        csv_path: Path to OHLCV CSV file
        
    Returns:
        DataFrame with OHLCV data
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ["close", "low", "high"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    
    return df


def analyze_limit_touches(
    df: pd.DataFrame,
    tail: int = 2000,
    thresholds_bps: list[int] = None,
) -> pd.DataFrame:
    """
    Analyze how often price touches LIMIT order levels at various thresholds.
    
    For each bar, we check:
    - buy_touch: Did the low touch or go below close * (1 - threshold)?
    - sell_touch: Did the high touch or go above close * (1 + threshold)?
    
    Args:
        df: OHLCV DataFrame with columns: close, low, high
        tail: Number of most recent bars to analyze
        thresholds_bps: List of thresholds in basis points (1 bps = 0.01%)
        
    Returns:
        DataFrame with columns: bps, buy_touch_rate, sell_touch_rate
    """
    if thresholds_bps is None:
        thresholds_bps = [1, 2, 5, 10, 15, 20, 25, 30, 40]
    
    # Take last N bars
    df_tail = df.tail(tail).copy()
    
    if len(df_tail) == 0:
        raise ValueError("No data available after applying tail filter")
    
    # Remove rows with invalid prices
    df_tail = df_tail[
        (df_tail["close"] > 0) & 
        (df_tail["low"] > 0) & 
        (df_tail["high"] > 0)
    ].copy()
    
    if len(df_tail) == 0:
        raise ValueError("No valid price data available")
    
    results = []
    
    for thr_bps in thresholds_bps:
        # Convert basis points to fraction (e.g., 10 bps = 0.001)
        threshold_frac = thr_bps * 1e-4
        
        # Calculate LIMIT levels based on previous close
        # For buy LIMIT: we want to buy below current price
        buy_limit = df_tail["close"] * (1 - threshold_frac)
        
        # For sell LIMIT: we want to sell above current price
        sell_limit = df_tail["close"] * (1 + threshold_frac)
        
        # Check if low touched or went below buy LIMIT
        buy_touches = (df_tail["low"] <= buy_limit).astype(int)
        buy_touch_rate = buy_touches.mean()
        
        # Check if high touched or went above sell LIMIT
        sell_touches = (df_tail["high"] >= sell_limit).astype(int)
        sell_touch_rate = sell_touches.mean()
        
        results.append({
            "bps": thr_bps,
            "buy_touch_rate": buy_touch_rate,
            "sell_touch_rate": sell_touch_rate,
        })
    
    return pd.DataFrame(results)


def print_results_table(results: pd.DataFrame) -> None:
    """Print results as a formatted table."""
    print("\n" + "=" * 60)
    print("LIMIT TOUCH ANALYSIS RESULTS")
    print("=" * 60)
    print()
    print(f"{'Threshold':<12} {'Buy Touch %':<15} {'Sell Touch %':<15}")
    print(f"{'(bps)':<12} {'(low <= limit)':<15} {'(high >= limit)':<15}")
    print("-" * 60)
    
    for _, row in results.iterrows():
        bps = int(row["bps"])
        buy_rate = row["buy_touch_rate"] * 100
        sell_rate = row["sell_touch_rate"] * 100
        print(f"{bps:<12} {buy_rate:<15.2f} {sell_rate:<15.2f}")
    
    print("=" * 60)
    print()


def save_results_csv(results: pd.DataFrame, output_path: Path) -> None:
    """Save results to CSV file."""
    results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze LIMIT order touch probability at various thresholds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to OHLCV CSV file (columns: timestamp, open, high, low, close, volume)",
    )
    
    parser.add_argument(
        "--tail",
        type=int,
        default=2000,
        help="Number of most recent bars to analyze (default: 2000)",
    )
    
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        help="Optional: Save results to CSV file",
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        return 1
    
    try:
        # Load data
        print(f"Loading OHLCV data from: {args.csv}")
        df = load_ohlcv(args.csv)
        print(f"Loaded {len(df)} bars")
        
        # Analyze
        print(f"Analyzing last {args.tail} bars...")
        results = analyze_limit_touches(df, tail=args.tail)
        
        # Display results
        print_results_table(results)
        
        # Save to CSV if requested
        if args.out:
            save_results_csv(results, args.out)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
