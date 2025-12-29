#!/usr/bin/env python3
"""
Comprehensive sanity check for derivatives data.

Validates:
- Monotonic UTC DatetimeIndex for all series
- Expected step sizes after resampling to 1h
- Missingness report per series
- Overlap intersection window across all required series

Example:
  python scripts/data/check_derivatives_sanity.py \
    --symbols BTCUSDT ETHUSDT \
    --start 2024-01-01 --end 2024-10-01 \
    --data-dir data/raw

This checks:
- data/raw/spot/{SYMBOL}_1h.csv.gz
- data/raw/futures/{SYMBOL}_funding.csv.gz
- data/raw/futures/{SYMBOL}_oi.csv.gz
- data/raw/futures/{SYMBOL}_mark_1h.csv.gz
- data/raw/futures/{SYMBOL}_basis.csv.gz (optional)
"""

from __future__ import annotations

import argparse
import gzip
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def ms_to_utc(ms: int) -> str:
    """Convert milliseconds to UTC string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_date_utc(s: str) -> int:
    """Parse YYYY-MM-DD or ISO datetime into UTC milliseconds."""
    s = s.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    if s.endswith("Z"):
        s = s[:-1]
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(f"Unsupported date format: {s}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * 1000)


def load_csv_gz(path: Path) -> Optional[pd.DataFrame]:
    """Load gzipped CSV and return DataFrame with timestamp index."""
    if not path.exists():
        return None
    
    try:
        df = pd.read_csv(path, compression="gzip")
        if "timestamp" not in df.columns:
            print(f"  ERROR: No 'timestamp' column in {path}")
            return None
        
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64")
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Convert to datetime index
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        return df
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return None


def check_monotonic(df: pd.DataFrame, name: str) -> bool:
    """Check if timestamps are strictly monotonic increasing."""
    if df is None or len(df) == 0:
        return False
    
    is_monotonic = df.index.is_monotonic_increasing
    has_duplicates = df.index.has_duplicates
    
    print(f"  {name}:")
    print(f"    Monotonic increasing: {is_monotonic}")
    print(f"    Has duplicates: {has_duplicates}")
    
    return is_monotonic and not has_duplicates


def check_step_sizes(df: pd.DataFrame, name: str, expected_step_ms: int = 3600000) -> Tuple[int, int]:
    """
    Check step sizes in the time series.
    
    Returns (num_correct_steps, num_gaps)
    """
    if df is None or len(df) < 2:
        return 0, 0
    
    # Calculate time differences in milliseconds
    time_diffs_ms = df.index.to_series().diff().dt.total_seconds() * 1000
    time_diffs_ms = time_diffs_ms.dropna()
    
    correct_steps = (time_diffs_ms == expected_step_ms).sum()
    total_steps = len(time_diffs_ms)
    gaps = total_steps - correct_steps
    
    print(f"  {name}:")
    print(f"    Expected step: {expected_step_ms} ms ({expected_step_ms / 3600000:.1f}h)")
    print(f"    Correct steps: {correct_steps}/{total_steps}")
    print(f"    Gaps/irregular steps: {gaps}")
    
    if gaps > 0 and gaps <= 10:
        # Show examples of irregular steps
        irregular = time_diffs_ms[time_diffs_ms != expected_step_ms].head(5)
        print(f"    Example irregular steps (ms): {irregular.tolist()}")
    
    return correct_steps, gaps


def check_missingness(df: pd.DataFrame, name: str, start_ms: int, end_ms: int, interval_ms: int = 3600000) -> float:
    """
    Check for missing data points.
    
    Returns percentage of missing data.
    """
    if df is None:
        print(f"  {name}: NO DATA")
        return 100.0
    
    expected_count = (end_ms - start_ms) // interval_ms
    actual_count = len(df)
    missing_count = max(0, expected_count - actual_count)
    missing_pct = (missing_count / expected_count * 100) if expected_count > 0 else 0
    
    print(f"  {name}:")
    print(f"    Expected records: {expected_count}")
    print(f"    Actual records: {actual_count}")
    print(f"    Missing: {missing_count} ({missing_pct:.2f}%)")
    print(f"    Data range: {ms_to_utc(int(df.index[0].timestamp() * 1000))} to {ms_to_utc(int(df.index[-1].timestamp() * 1000))}")
    
    return missing_pct


def check_overlap(dataframes: Dict[str, pd.DataFrame]) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Find the overlapping time window across all dataframes.
    
    Returns (overlap_start, overlap_end)
    """
    if not dataframes:
        return None, None
    
    valid_dfs = {k: v for k, v in dataframes.items() if v is not None and len(v) > 0}
    
    if not valid_dfs:
        return None, None
    
    # Find common overlap
    start_times = [df.index[0] for df in valid_dfs.values()]
    end_times = [df.index[-1] for df in valid_dfs.values()]
    
    overlap_start = max(start_times)
    overlap_end = min(end_times)
    
    print("\nOverlap Analysis:")
    for name, df in valid_dfs.items():
        print(f"  {name}: {df.index[0]} to {df.index[-1]}")
    
    if overlap_start <= overlap_end:
        overlap_duration_hours = (overlap_end - overlap_start).total_seconds() / 3600
        print(f"\n  INTERSECTION WINDOW:")
        print(f"    Start: {overlap_start}")
        print(f"    End: {overlap_end}")
        print(f"    Duration: {overlap_duration_hours:.1f} hours ({overlap_duration_hours/24:.1f} days)")
    else:
        print(f"\n  WARNING: No overlapping period found!")
        overlap_start = None
        overlap_end = None
    
    return overlap_start, overlap_end


def check_value_ranges(df: pd.DataFrame, name: str, column: str, expected_range: Optional[Tuple[float, float]] = None):
    """Check value ranges for sanity."""
    if df is None or column not in df.columns:
        return
    
    values = df[column].dropna()
    if len(values) == 0:
        return
    
    print(f"  {name} - {column}:")
    print(f"    Min: {values.min():.8f}")
    print(f"    Max: {values.max():.8f}")
    print(f"    Mean: {values.mean():.8f}")
    print(f"    NaN count: {df[column].isna().sum()}")
    
    if expected_range:
        min_val, max_val = expected_range
        out_of_range = ((values < min_val) | (values > max_val)).sum()
        if out_of_range > 0:
            print(f"    WARNING: {out_of_range} values outside expected range [{min_val}, {max_val}]")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Comprehensive sanity check for derivatives data"
    )
    ap.add_argument("--symbols", nargs="+", required=True, help="Symbols to check (e.g., BTCUSDT ETHUSDT)")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--data-dir", default="data/raw", help="Base data directory (default: data/raw)")
    ap.add_argument("--skip-basis", action="store_true", help="Skip basis check (optional data)")
    args = ap.parse_args()
    
    start_ms = parse_date_utc(args.start)
    end_ms = parse_date_utc(args.end)
    data_dir = Path(args.data_dir)
    
    print("=" * 80)
    print("DERIVATIVES DATA SANITY CHECK")
    print("=" * 80)
    print(f"Period: {args.start} to {args.end}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Data directory: {data_dir}")
    print("=" * 80)
    
    all_pass = True
    
    for symbol in args.symbols:
        print(f"\n{'=' * 80}")
        print(f"Checking {symbol}")
        print(f"{'=' * 80}")
        
        # Define file paths
        spot_path = data_dir / "spot" / f"{symbol}_1h.csv.gz"
        funding_path = data_dir / "futures" / f"{symbol}_funding.csv.gz"
        oi_path = data_dir / "futures" / f"{symbol}_oi.csv.gz"
        mark_path = data_dir / "futures" / f"{symbol}_mark_1h.csv.gz"
        basis_path = data_dir / "futures" / f"{symbol}_basis.csv.gz"
        
        # Load data
        print("\n1. LOADING DATA")
        print("-" * 80)
        spot_df = load_csv_gz(spot_path)
        funding_df = load_csv_gz(funding_path)
        oi_df = load_csv_gz(oi_path)
        mark_df = load_csv_gz(mark_path)
        basis_df = None if args.skip_basis else load_csv_gz(basis_path)
        
        dataframes = {
            "spot": spot_df,
            "funding": funding_df,
            "open_interest": oi_df,
            "mark": mark_df,
        }
        if not args.skip_basis and basis_df is not None:
            dataframes["basis"] = basis_df
        
        # Check monotonicity
        print("\n2. MONOTONICITY CHECK")
        print("-" * 80)
        for name, df in dataframes.items():
            if df is not None:
                if not check_monotonic(df, name):
                    all_pass = False
        
        # Check step sizes (expecting 1h = 3600000 ms after resampling)
        # Exception: funding rates are native 8h intervals
        print("\n3. STEP SIZE CHECK")
        print("-" * 80)
        for name, df in dataframes.items():
            if df is not None:
                if name == "funding":
                    # Funding rates are published every 8 hours
                    correct, gaps = check_step_sizes(df, name, expected_step_ms=8*3600000)
                else:
                    correct, gaps = check_step_sizes(df, name, expected_step_ms=3600000)
                    if gaps > len(df) * 0.01:  # More than 1% gaps
                        print(f"    WARNING: High gap rate for {name}")
        
        # Check missingness
        print("\n4. MISSINGNESS REPORT")
        print("-" * 80)
        for name, df in dataframes.items():
            if name == "funding":
                # Funding rates are published every 8 hours
                missing_pct = check_missingness(df, name, start_ms, end_ms, interval_ms=8*3600000)
            else:
                missing_pct = check_missingness(df, name, start_ms, end_ms, interval_ms=3600000)
            
            if missing_pct > 5.0 and name != "funding":
                print(f"    WARNING: High missingness for {name}: {missing_pct:.2f}%")
                all_pass = False
        
        # Check overlap
        print("\n5. OVERLAP INTERSECTION")
        print("-" * 80)
        overlap_start, overlap_end = check_overlap(dataframes)
        if overlap_start is None or overlap_end is None:
            print("  ERROR: No valid overlap found!")
            all_pass = False
        else:
            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
            if overlap_hours < 24 * 30:  # Less than 30 days
                print(f"  WARNING: Overlap window is short ({overlap_hours/24:.1f} days)")
        
        # Check value ranges
        print("\n6. VALUE RANGE CHECKS")
        print("-" * 80)
        if spot_df is not None:
            check_value_ranges(spot_df, "spot", "close")
        if funding_df is not None:
            # Funding rates typically between -1% to +1%
            check_value_ranges(funding_df, "funding", "fundingRate", expected_range=(-0.01, 0.01))
        if oi_df is not None:
            check_value_ranges(oi_df, "open_interest", "sumOpenInterest")
        if mark_df is not None:
            check_value_ranges(mark_df, "mark", "close")
        if basis_df is not None:
            check_value_ranges(basis_df, "basis", "basis")
        
        print("\n" + "=" * 80)
    
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ ALL SANITY CHECKS PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME SANITY CHECKS FAILED - Review warnings above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
