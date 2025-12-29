#!/usr/bin/env python3
"""
Generate mock derivatives data for testing purposes.

This creates sample data files in the correct format to test the sanity check script
and validate the data protocol without requiring actual API access.

Example:
  python scripts/data/generate_mock_derivatives_data.py \
    --symbols BTCUSDT ETHUSDT \
    --start 2024-01-01 --end 2024-10-01 \
    --data-dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import random


def parse_date_utc(s: str) -> int:
    """Parse YYYY-MM-DD or ISO datetime into UTC milliseconds."""
    s = s.strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    raise ValueError(f"Unsupported date format: {s}")


def generate_spot_klines(symbol: str, start_ms: int, end_ms: int, base_price: float) -> List:
    """Generate mock spot klines data."""
    data = []
    interval_ms = 3600000  # 1 hour
    
    t = start_ms
    price = base_price
    
    while t < end_ms:
        # Random walk with small steps
        price_change = random.uniform(-0.02, 0.02) * price
        price += price_change
        
        open_price = price
        high_price = price * (1 + random.uniform(0, 0.01))
        low_price = price * (1 - random.uniform(0, 0.01))
        close_price = price * random.uniform(0.995, 1.005)
        volume = random.uniform(100, 1000)
        
        data.append({
            'timestamp': t,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
        t += interval_ms
    
    return data


def generate_funding_rates(symbol: str, start_ms: int, end_ms: int) -> List:
    """Generate mock funding rate data (every 8 hours)."""
    data = []
    interval_ms = 8 * 3600000  # 8 hours
    
    t = start_ms
    # Align to 00:00, 08:00, 16:00 UTC
    t = t - (t % interval_ms)
    
    while t < end_ms:
        funding_rate = random.uniform(-0.0005, 0.0005)
        data.append({
            'timestamp': t,
            'fundingRate': funding_rate,
            'symbol': symbol
        })
        t += interval_ms
    
    return data


def generate_mark_klines(symbol: str, start_ms: int, end_ms: int, base_price: float) -> List:
    """Generate mock mark price klines."""
    data = []
    interval_ms = 3600000  # 1 hour
    
    t = start_ms
    price = base_price * 1.0001  # Slightly above spot
    
    while t < end_ms:
        price_change = random.uniform(-0.02, 0.02) * price
        price += price_change
        
        open_price = price
        high_price = price * (1 + random.uniform(0, 0.01))
        low_price = price * (1 - random.uniform(0, 0.01))
        close_price = price * random.uniform(0.995, 1.005)
        
        data.append({
            'timestamp': t,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price
        })
        
        price = close_price
        t += interval_ms
    
    return data


def generate_open_interest(symbol: str, start_ms: int, end_ms: int, base_oi: float) -> List:
    """Generate mock open interest data."""
    data = []
    interval_ms = 3600000  # 1 hour
    
    t = start_ms
    oi = base_oi
    
    while t < end_ms:
        # Slowly varying open interest
        oi_change = random.uniform(-0.01, 0.01) * oi
        oi += oi_change
        oi = max(oi, base_oi * 0.5)  # Keep positive
        
        oi_value = oi * random.uniform(40000, 50000)  # Approximate USDT value
        
        data.append({
            'timestamp': t,
            'sumOpenInterest': oi,
            'sumOpenInterestValue': oi_value,
            'symbol': symbol
        })
        
        t += interval_ms
    
    return data


def generate_basis(symbol: str, start_ms: int, end_ms: int, base_price: float) -> List:
    """Generate mock basis data."""
    data = []
    interval_ms = 3600000  # 1 hour
    
    t = start_ms
    
    while t < end_ms:
        # Basis is usually small and can be positive or negative
        basis = random.uniform(-20, 20)
        basis_rate = basis / base_price
        annualized_basis_rate = basis_rate * 365 * 24  # Rough annualization
        
        data.append({
            'timestamp': t,
            'basis': basis,
            'basisRate': basis_rate,
            'annualizedBasisRate': annualized_basis_rate,
            'symbol': symbol
        })
        
        t += interval_ms
    
    return data


def write_csv_gz(path: Path, data: List[dict], columns: List[str]) -> None:
    """Write data to gzipped CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with gzip.open(path, 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in data:
            # Format floats to 8 decimal places
            formatted_row = {}
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    formatted_row[col] = f"{val:.8f}"
                else:
                    formatted_row[col] = val
            writer.writerow(formatted_row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate mock derivatives data for testing"
    )
    ap.add_argument("--symbols", nargs="+", required=True, help="Symbols to generate (e.g., BTCUSDT ETHUSDT)")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--data-dir", default="data/raw", help="Base data directory (default: data/raw)")
    args = ap.parse_args()
    
    start_ms = parse_date_utc(args.start)
    end_ms = parse_date_utc(args.end)
    data_dir = Path(args.data_dir)
    
    # Base prices for different symbols
    base_prices = {
        'BTCUSDT': 42000.0,
        'ETHUSDT': 2200.0,
        'BNBUSDT': 300.0,
    }
    
    base_oi = {
        'BTCUSDT': 100000.0,
        'ETHUSDT': 500000.0,
        'BNBUSDT': 50000.0,
    }
    
    print(f"Generating mock data for period: {args.start} to {args.end}")
    
    for symbol in args.symbols:
        print(f"\nGenerating data for {symbol}...")
        
        base_price = base_prices.get(symbol, 1000.0)
        base_oi_val = base_oi.get(symbol, 10000.0)
        
        # Generate spot klines
        spot_data = generate_spot_klines(symbol, start_ms, end_ms, base_price)
        spot_path = data_dir / "spot" / f"{symbol}_1h.csv.gz"
        write_csv_gz(spot_path, spot_data, ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        print(f"  Created {spot_path} ({len(spot_data)} records)")
        
        # Generate funding rates
        funding_data = generate_funding_rates(symbol, start_ms, end_ms)
        funding_path = data_dir / "futures" / f"{symbol}_funding.csv.gz"
        write_csv_gz(funding_path, funding_data, ['timestamp', 'fundingRate', 'symbol'])
        print(f"  Created {funding_path} ({len(funding_data)} records)")
        
        # Generate mark price klines
        mark_data = generate_mark_klines(symbol, start_ms, end_ms, base_price)
        mark_path = data_dir / "futures" / f"{symbol}_mark_1h.csv.gz"
        write_csv_gz(mark_path, mark_data, ['timestamp', 'open', 'high', 'low', 'close'])
        print(f"  Created {mark_path} ({len(mark_data)} records)")
        
        # Generate open interest
        oi_data = generate_open_interest(symbol, start_ms, end_ms, base_oi_val)
        oi_path = data_dir / "futures" / f"{symbol}_oi.csv.gz"
        write_csv_gz(oi_path, oi_data, ['timestamp', 'sumOpenInterest', 'sumOpenInterestValue', 'symbol'])
        print(f"  Created {oi_path} ({len(oi_data)} records)")
        
        # Generate basis
        basis_data = generate_basis(symbol, start_ms, end_ms, base_price)
        basis_path = data_dir / "futures" / f"{symbol}_basis.csv.gz"
        write_csv_gz(basis_path, basis_data, ['timestamp', 'basis', 'basisRate', 'annualizedBasisRate', 'symbol'])
        print(f"  Created {basis_path} ({len(basis_data)} records)")
    
    # Generate futures exchange info
    metadata_dir = data_dir.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    exchange_info = {
        "fetchTime": int(datetime.now(timezone.utc).timestamp() * 1000),
        "timezone": "UTC",
        "contracts": []
    }
    
    for symbol in args.symbols:
        contract = {
            "symbol": symbol,
            "pair": symbol,
            "contractType": "PERPETUAL",
            "deliveryDate": None,
            "onboardDate": 1609459200000,
            "status": "TRADING",
            "baseAsset": symbol[:-4],
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "pricePrecision": 2,
            "quantityPrecision": 3
        }
        exchange_info["contracts"].append(contract)
    
    exchange_info_path = metadata_dir / "futures_exchangeInfo.json"
    with open(exchange_info_path, 'w') as f:
        json.dump(exchange_info, f, indent=2)
    print(f"\nCreated {exchange_info_path}")
    
    print("\nMock data generation complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
