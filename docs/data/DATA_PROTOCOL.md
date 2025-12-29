# Data Protocol

This document defines the canonical data storage and processing conventions for theta-bot's derivatives data acquisition system.

## Core Principles

### 1. UTC-Only Timestamps
- **ALL** timestamps MUST be in UTC
- NO local timezone conversions
- Timestamps represent canonical close-time for each interval
- Store as millisecond epoch integers for compatibility with Binance API

### 2. Canonical Close-Time Grid
- All time-series data aligns to interval close times
- For 1h interval: data point at 2024-01-01 01:00:00 represents [00:00:00, 01:00:00)
- Ensures consistent temporal alignment across all data sources

### 3. Baseline Interval: 1 Hour
- Primary analysis interval is **1h**
- All data resampled to 1h for unified analysis
- Higher frequency data (e.g., 1m, 5m, 15m) can be aggregated to 1h
- Lower frequency data (e.g., 4h, 8h, 1d) forward-filled to 1h

## Data Sources and Endpoints

### Spot Market Data
- **Klines**: `/api/v3/klines`
  - OHLCV candlestick data
  - Available intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
  - Public endpoint (no API key required)

### Futures Market Data

#### Funding Rate History
- **Endpoint**: `/fapi/v1/fundingRate`
- **Native Interval**: 8 hours (00:00, 08:00, 16:00 UTC)
- **Resampling Rule**: Forward-fill to 1h
  - Funding rate at 00:00 applies to hours 00:00-07:59
  - Funding rate at 08:00 applies to hours 08:00-15:59
  - Funding rate at 16:00 applies to hours 16:00-23:59
- **Columns**: timestamp, fundingRate, symbol
- **Public endpoint** (no API key required for historical data)

#### Mark Price Klines
- **Endpoint**: `/fapi/v1/markPriceKlines`
- **Intervals**: Same as spot klines (1m, 5m, 15m, 1h, etc.)
- **Resampling Rule**: Standard OHLCV aggregation to 1h if needed
- **Columns**: timestamp, open, high, low, close
- **Note**: Mark price is used for funding rate calculation and liquidations

#### Premium Index Klines
- **Endpoint**: `/fapi/v1/premiumIndex`
- **Use**: Current mark price and funding info
- **Historical**: Use `/fapi/v1/premiumIndexKlines` for historical premium index

#### Continuous Klines
- **Endpoint**: `/fapi/v1/continuousKlines`
- **Use**: For perpetual futures without rollover gaps
- **Parameters**: contractType (PERPETUAL, CURRENT_QUARTER, NEXT_QUARTER)

#### Open Interest History
- **Endpoint**: `/futures/data/openInterestHist`
- **Native Interval**: Varies (typically 5m or 15m)
- **Resampling Rule**: Use last value in each 1h window
- **Columns**: timestamp, sumOpenInterest, sumOpenInterestValue, symbol
- **Note**: Total open interest in base asset and USDT value

#### Basis
- **Endpoint**: `/futures/data/basis`
- **Definition**: basis = futures_price - index_price
- **Native Interval**: 5m or computed
- **Resampling Rule**: Use last value in each 1h window
- **Fallback**: If endpoint unavailable, compute as `mark_price - spot_price`
- **Columns**: timestamp, basis, basisRate (basis/spot), annualizedBasisRate

#### Futures Exchange Info
- **Endpoint**: `/fapi/v1/exchangeInfo`
- **Purpose**: Metadata for contract types, delivery dates, expiry features
- **Fields**: symbol, contractType, deliveryDate, onboardDate, status
- **Storage**: JSON format in metadata directory
- **Update Frequency**: Daily or as needed for contract rollovers

## File Naming Convention

### Directory Structure
```
data/
├── raw/
│   ├── spot/
│   │   ├── {SYMBOL}_{INTERVAL}.csv.gz
│   │   └── ...
│   └── futures/
│       ├── {SYMBOL}_funding.csv.gz
│       ├── {SYMBOL}_oi.csv.gz
│       ├── {SYMBOL}_mark_{INTERVAL}.csv.gz
│       ├── {SYMBOL}_basis.csv.gz
│       └── ...
├── processed/
│   └── {processing outputs}
└── metadata/
    ├── futures_exchangeInfo.json
    └── ...
```

### Naming Rules
1. **Spot Klines**: `{SYMBOL}_{INTERVAL}.csv.gz`
   - Example: `BTCUSDT_1h.csv.gz`, `ETHUSDT_15m.csv.gz`

2. **Futures Funding**: `{SYMBOL}_funding.csv.gz`
   - Example: `BTCUSDT_funding.csv.gz`

3. **Futures Open Interest**: `{SYMBOL}_oi.csv.gz`
   - Example: `BTCUSDT_oi.csv.gz`

4. **Futures Mark Price Klines**: `{SYMBOL}_mark_{INTERVAL}.csv.gz`
   - Example: `BTCUSDT_mark_1h.csv.gz`

5. **Futures Basis**: `{SYMBOL}_basis.csv.gz`
   - Example: `BTCUSDT_basis.csv.gz`

6. **Metadata**: `futures_exchangeInfo.json`
   - Single file containing all futures contract metadata

## CSV Format Specifications

### Standard Klines (Spot and Mark Price)
```csv
timestamp,open,high,low,close,volume
1704067200000,42150.50000000,42285.30000000,42100.00000000,42250.75000000,1234.56780000
```

### Funding Rate
```csv
timestamp,fundingRate,symbol
1704067200000,0.00010000,BTCUSDT
```

### Open Interest
```csv
timestamp,sumOpenInterest,sumOpenInterestValue,symbol
1704067200000,12345.67800000,520000000.00000000,BTCUSDT
```

### Basis
```csv
timestamp,basis,basisRate,annualizedBasisRate,symbol
1704067200000,15.50000000,0.00036700,0.13414200,BTCUSDT
```

## Resampling Rules Detail

### Funding Rate (8h → 1h)
```python
# Forward-fill method
df_funding_1h = df_funding_8h.resample('1h').ffill()

# Explicit logic:
# - Funding at 00:00 UTC → apply to 00:00-07:59
# - Funding at 08:00 UTC → apply to 08:00-15:59
# - Funding at 16:00 UTC → apply to 16:00-23:59
```

### Open Interest (5m → 1h)
```python
# Last observation in window
df_oi_1h = df_oi_5m.resample('1h').last()
```

### Mark Price Klines (5m → 1h)
```python
# Standard OHLCV aggregation
df_mark_1h = df_mark_5m.resample('1h').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'  # if available
})
```

### Basis (5m → 1h)
```python
# Last observation in window
df_basis_1h = df_basis_5m.resample('1h').last()
```

## Data Quality Requirements

### Monotonicity
- Timestamps MUST be strictly increasing
- No duplicate timestamps allowed
- Use `df.index.is_monotonic_increasing` to verify

### Completeness
- Report missing intervals explicitly
- Calculate expected vs actual record counts
- Generate missingness summary per series

### Temporal Alignment
- All 1h-resampled series should share the same timestamp index
- Calculate overlap intersection window across all series
- Report earliest common start and latest common end

### Range Validation
- Funding rates: typically -0.01 to +0.01 (±1%)
- Open interest: strictly positive
- Basis: can be positive or negative
- Mark price: should be close to spot price (within ±10% typically)

## Sanity Check Requirements

The `check_derivatives_sanity.py` script MUST verify:

1. **Monotonic UTC DatetimeIndex**
   - All series have strictly increasing timestamps
   - No duplicates

2. **Expected Step Sizes**
   - After resampling to 1h: step = 3600000 ms (1 hour)
   - Identify and report any gaps or irregular steps

3. **Missingness Report**
   - Expected records: `(end_date - start_date) / 1h`
   - Actual records per series
   - Missing percentage and gap locations

4. **Overlap Intersection**
   - Common time window across ALL required series
   - Report intersection start/end dates
   - Warn if intersection is too small for analysis

## Usage Examples

### Download Spot Klines
```bash
python scripts/data/binance_download_klines.py \
  --symbol BTCUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/spot/BTCUSDT_1h.csv.gz
```

### Download Funding History
```bash
python scripts/data/binance_download_funding.py \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/futures/BTCUSDT_funding.csv.gz
```

### Download Mark Price Klines
```bash
python scripts/data/binance_download_mark_klines.py \
  --symbol BTCUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/futures/BTCUSDT_mark_1h.csv.gz
```

### Download Open Interest History
```bash
python scripts/data/binance_download_open_interest_hist.py \
  --symbol BTCUSDT \
  --period 1h \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/futures/BTCUSDT_oi.csv.gz
```

### Fetch Futures Metadata
```bash
python scripts/data/binance_fetch_futures_exchangeinfo.py \
  --out data/metadata/futures_exchangeInfo.json
```

### Run Sanity Checks
```bash
python scripts/data/check_derivatives_sanity.py \
  --symbols BTCUSDT ETHUSDT \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --data-dir data/raw
```

## Notes

- **API Rate Limits**: Binance has rate limits. Scripts include throttling (0.2s default between requests)
- **No API Keys Required**: All endpoints used are public (historical data)
- **Error Handling**: Scripts retry with exponential backoff on transient failures
- **Data Persistence**: All files are compressed with gzip for efficient storage
- **Reproducibility**: Given the same symbol and date range, scripts should produce identical output
