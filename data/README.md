# Data Directory

This directory contains datasets used for reproducible evaluation of trading models.

## BTCUSDT_1H_sample.csv.gz

**Description:** Synthetic but realistic BTCUSDT 1-hour candlestick data for evaluation purposes.

**Details:**
- **Symbol:** BTCUSDT (Bitcoin/USDT)
- **Timeframe:** 1 hour (1H) candles
- **Date Range:** 2024-06-01 to 2024-11-27 (~6 months, 180 days)
- **Bars:** 4,320 hourly candles
- **Size:** ~210 KB (compressed)

**Columns:**
- `timestamp`: UTC timestamp (ISO format with timezone)
- `open`: Opening price for the hour
- `high`: Highest price during the hour
- `low`: Lowest price during the hour
- `close`: Closing price for the hour
- `volume`: Trading volume for the hour

**Data Characteristics:**
- Synthetic data generated with realistic market patterns:
  - Trend components (slow drift)
  - Multiple oscillation cycles (weekly, monthly, short-term)
  - Random walk component (Brownian motion)
  - Realistic OHLC relationships and volatility
  - Volume correlated with price volatility

**Purpose:**
This dataset is used for reproducible offline evaluation of trading models, particularly comparing baseline models against the Dual-Stream (Theta + Mellin) architecture. It allows for consistent benchmarking without requiring internet access or real market data dependencies.

**Usage:**
```python
import pandas as pd

# Load the data
df = pd.read_csv("data/BTCUSDT_1H_sample.csv.gz", index_col=0, parse_dates=True)
print(df.head())
```

**Notes:**
- This is synthetic data for evaluation and testing purposes only.
- Not intended for production trading or real market analysis.
- The data exhibits realistic statistical properties for testing model behavior.
- Committed to the repository for reproducible evaluation across environments.
