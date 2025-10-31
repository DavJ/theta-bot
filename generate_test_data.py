#!/usr/bin/env python3
"""
Generate synthetic test data for theta experiments.
"""
import numpy as np
import pandas as pd
import os

def generate_synthetic_prices(n_samples=2000, seed=42):
    """
    Generate synthetic price data with:
    - Trend
    - Multiple periodic components
    - Noise
    """
    np.random.seed(seed)
    
    t = np.arange(n_samples)
    
    # Trend
    trend = 1000 + 0.5 * t
    
    # Multiple periodic components (simulating market cycles)
    cycle1 = 50 * np.sin(2 * np.pi * t / 100)  # Long cycle
    cycle2 = 20 * np.sin(2 * np.pi * t / 30)   # Medium cycle
    cycle3 = 10 * np.sin(2 * np.pi * t / 10)   # Short cycle
    
    # Random walk component
    random_walk = np.cumsum(np.random.randn(n_samples) * 5)
    
    # Combine
    prices = trend + cycle1 + cycle2 + cycle3 + random_walk
    
    # Ensure positive prices
    prices = np.abs(prices) + 100
    
    return prices

def main():
    # Create output directory
    os.makedirs('test_data', exist_ok=True)
    
    # Generate data
    print("Generating synthetic price data...")
    prices = generate_synthetic_prices(n_samples=2000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=len(prices), freq='H'),
        'close': prices,
        'open': prices * (1 + np.random.randn(len(prices)) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(len(prices)) * 0.002)),
        'low': prices * (1 - np.abs(np.random.randn(len(prices)) * 0.002)),
        'volume': np.random.uniform(1000, 10000, len(prices))
    })
    
    # Save
    output_path = 'test_data/synthetic_prices.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(f"Samples: {len(df)}")
    print(f"Price range: [{df['close'].min():.2f}, {df['close'].max():.2f}]")

if __name__ == '__main__':
    main()
