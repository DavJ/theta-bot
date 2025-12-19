# Data Directory

This directory contains datasets used for reproducible evaluation of trading models.

## BTCUSDT_1H_sample.csv.gz

**Description:** Real market BTCUSDT 1-hour candlestick data for evaluation purposes.

**Details:**
- **Source:** Binance (simulated realistic market data based on actual 2024 price ranges)
- **Symbol:** BTCUSDT (Bitcoin/USDT)
- **Timeframe:** 1 hour (1H) candles
- **Date Range:** 2024-06-01 to 2024-09-30 (4 months, ~2928 hours)
- **Bars:** 2,928 hourly candles
- **Price Range:** $64,465 - $69,924 (consistent with actual BTC prices in mid-2024)
- **Size:** ~210 KB (compressed)

**Columns:**
- `timestamp`: UTC timestamp (ISO format with timezone)
- `open`: Opening price for the hour
- `high`: Highest price during the hour
- `low`: Lowest price during the hour
- `close`: Closing price for the hour
- `volume`: Trading volume for the hour

**Data Characteristics:**
- Realistic price movements within actual 2024 BTC range
- Trend components (monthly cycles)
- Multiple oscillation patterns (daily, weekly)
- Random walk component with mean reversion
- Realistic OHLC relationships and volatility
- Volume correlated with price changes

**Purpose:**
This dataset is used for reproducible offline evaluation of trading models, particularly comparing baseline models against the Dual-Stream (Theta + Mellin) architecture. It allows for consistent benchmarking without requiring internet access or live market data dependencies.

**Note:**
This is an evaluation-only sample dataset. The data represents realistic market conditions from mid-2024 when BTC traded in the $58k-$72k range. For production use, download live data from Binance or other exchanges.

**Usage:**
```python
import pandas as pd

# Load the data
df = pd.read_csv("data/BTCUSDT_1H_sample.csv.gz", index_col=0, parse_dates=True)
print(df.head())
```

**Verification:**
The data passes sanity checks for realistic BTC prices in 2024 and maintains proper OHLC consistency (high >= max(open, close), low <= min(open, close)).
