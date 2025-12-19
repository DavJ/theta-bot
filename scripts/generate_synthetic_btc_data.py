#!/usr/bin/env python3
"""
Generate synthetic but realistic BTCUSDT 1H candlestick data for TESTING ONLY.

NOTE: This script generates SYNTHETIC data for unit tests and development.
      It should NOT be used for the main evaluation.
      The main evaluation must use: data/BTCUSDT_1H_real.csv.gz
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_btc_data(
    n_bars=4320,  # ~6 months of hourly data (180 days * 24 hours)
    initial_price=40000.0,
    seed=42,
):
    """
    Generate synthetic BTCUSDT 1H data with realistic characteristics:
    - Trend components
    - Multiple oscillation frequencies
    - Realistic volatility
    - Volume patterns
    """
    np.random.seed(seed)
    
    # Time array
    t = np.arange(n_bars)
    
    # Start date: 6 months ago from a reference point
    start_date = pd.Timestamp("2024-06-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start_date, periods=n_bars, freq="1h")
    
    # Price components:
    # 1. Trend (slow drift)
    trend = 0.00003 * t  # ~13% over 6 months
    
    # 2. Multiple oscillations (like market cycles)
    # Weekly cycle
    weekly_cycle = 0.03 * np.sin(2 * np.pi * t / (7 * 24))
    # Monthly cycle
    monthly_cycle = 0.05 * np.sin(2 * np.pi * t / (30 * 24))
    # Shorter term fluctuations
    short_cycle = 0.02 * np.sin(2 * np.pi * t / (3 * 24))
    
    # 3. Random walk component (brownian motion)
    returns = np.random.randn(n_bars) * 0.015
    random_walk = np.cumsum(returns)
    random_walk = random_walk - np.mean(random_walk)  # Center it
    
    # Combine components
    log_price = (
        np.log(initial_price)
        + trend
        + weekly_cycle
        + monthly_cycle
        + short_cycle
        + random_walk
    )
    close = np.exp(log_price)
    
    # Generate OHLC from close
    # High/Low should bracket close realistically
    volatility = 0.008  # ~0.8% typical bar range
    high = close * (1 + np.abs(np.random.randn(n_bars) * volatility))
    low = close * (1 - np.abs(np.random.randn(n_bars) * volatility))
    
    # Open should be close to previous close
    open_price = np.concatenate([[close[0]], close[:-1]])
    # Add small random adjustment to open
    open_price = open_price * (1 + np.random.randn(n_bars) * 0.003)
    
    # Ensure OHLC consistency: high >= max(open, close), low <= min(open, close)
    for i in range(n_bars):
        max_oc = max(open_price[i], close[i])
        min_oc = min(open_price[i], close[i])
        if high[i] < max_oc:
            high[i] = max_oc * 1.001
        if low[i] > min_oc:
            low[i] = min_oc * 0.999
    
    # Generate volume (correlated with volatility)
    bar_volatility = (high - low) / close
    base_volume = 100.0  # Base volume
    volume = base_volume * (1 + 5 * bar_volatility) * np.exp(np.random.randn(n_bars) * 0.3)
    
    # Create DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    
    # Set timestamp as index
    df = df.set_index("timestamp")
    
    return df


def main():
    """Generate and save synthetic BTCUSDT data for testing purposes."""
    print("=" * 70)
    print("⚠️  WARNING: Generating SYNTHETIC data for TESTING ONLY")
    print("=" * 70)
    print("This script creates synthetic data for unit tests and development.")
    print("The main evaluation MUST use: data/BTCUSDT_1H_real.csv.gz")
    print("=" * 70 + "\n")
    
    print("Generating synthetic BTCUSDT 1H data...")
    df = generate_synthetic_btc_data()
    
    # Save to data directory with "_synthetic" suffix to avoid confusion
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "BTCUSDT_1H_synthetic_test.csv.gz"
    df.to_csv(output_path, compression="gzip")
    
    print(f"✓ Generated {len(df)} bars")
    print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
    print(f"✓ Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 70)
    print("⚠️  REMINDER: This is SYNTHETIC data for testing only!")
    print("Main evaluation uses: data/BTCUSDT_1H_real.csv.gz")
    print("=" * 70)
    
    # Show sample
    print("\nFirst few rows:")
    print(df.head())
    print("\nLast few rows:")
    print(df.tail())
    
    return df


if __name__ == "__main__":
    main()
