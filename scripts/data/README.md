# Data Download Scripts

This directory contains scripts for downloading and validating derivatives market data from Binance.

## Overview

These scripts implement the data acquisition protocol defined in [docs/data/DATA_PROTOCOL.md](../../docs/data/DATA_PROTOCOL.md).

All scripts are **public endpoint only** - no API keys required for historical data.

## Scripts

### Spot Market Data

#### `binance_download_klines.py`
Download spot market OHLCV klines.

```bash
python scripts/data/binance_download_klines.py \
  --symbol BTCUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/spot/BTCUSDT_1h.csv.gz
```

**Supported intervals:** 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

### Futures Market Data

#### `binance_download_funding.py`
Download futures funding rate history (published every 8 hours).

```bash
python scripts/data/binance_download_funding.py \
  --symbol BTCUSDT \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/futures/BTCUSDT_funding.csv.gz
```

**Output columns:** timestamp, fundingRate, symbol

#### `binance_download_mark_klines.py`
Download futures mark price klines.

```bash
python scripts/data/binance_download_mark_klines.py \
  --symbol BTCUSDT \
  --interval 1h \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/futures/BTCUSDT_mark_1h.csv.gz
```

**Output columns:** timestamp, open, high, low, close

#### `binance_download_open_interest_hist.py`
Download futures open interest history.

```bash
python scripts/data/binance_download_open_interest_hist.py \
  --symbol BTCUSDT \
  --period 1h \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/futures/BTCUSDT_oi.csv.gz
```

**Supported periods:** 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d

**Output columns:** timestamp, sumOpenInterest, sumOpenInterestValue, symbol

#### `binance_download_basis.py`
Download futures basis history (futures price - index price).

```bash
python scripts/data/binance_download_basis.py \
  --symbol BTCUSDT \
  --period 5m \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --out data/raw/futures/BTCUSDT_basis.csv.gz
```

**Note:** This endpoint may not be available for all symbols. If unavailable, compute basis from mark_price - spot_price.

**Output columns:** timestamp, basis, basisRate, annualizedBasisRate, symbol

#### `binance_fetch_futures_exchangeinfo.py`
Fetch futures exchange metadata (contract types, delivery dates, etc.).

```bash
python scripts/data/binance_fetch_futures_exchangeinfo.py \
  --out data/metadata/futures_exchangeInfo.json
```

Use `--full` to save the complete API response instead of extracted metadata.

### Data Validation

#### `check_derivatives_sanity.py`
Comprehensive sanity check for all derivatives data.

```bash
python scripts/data/check_derivatives_sanity.py \
  --symbols BTCUSDT ETHUSDT \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --data-dir data/raw
```

**Checks performed:**
1. Monotonic UTC DatetimeIndex (no duplicates, strictly increasing)
2. Expected step sizes (1h for most data, 8h for funding)
3. Missingness report per series
4. Overlap intersection window across all series
5. Value range validation

Use `--skip-basis` to skip basis data check (optional data).

### Testing & Development

#### `generate_mock_derivatives_data.py`
Generate mock data for testing without API access.

```bash
python scripts/data/generate_mock_derivatives_data.py \
  --symbols BTCUSDT ETHUSDT \
  --start 2024-01-01 \
  --end 2024-10-01 \
  --data-dir data/raw
```

This creates realistic-looking test data in the correct format for development and testing.

## Complete Workflow Example

Download all derivatives data for BTCUSDT and ETHUSDT:

```bash
# Set date range
START="2024-01-01"
END="2024-10-01"

# Download spot klines (1h interval)
for SYMBOL in BTCUSDT ETHUSDT; do
  python scripts/data/binance_download_klines.py \
    --symbol $SYMBOL --interval 1h \
    --start $START --end $END \
    --out data/raw/spot/${SYMBOL}_1h.csv.gz
done

# Download futures data
for SYMBOL in BTCUSDT ETHUSDT; do
  # Funding rates
  python scripts/data/binance_download_funding.py \
    --symbol $SYMBOL \
    --start $START --end $END \
    --out data/raw/futures/${SYMBOL}_funding.csv.gz
  
  # Mark price klines
  python scripts/data/binance_download_mark_klines.py \
    --symbol $SYMBOL --interval 1h \
    --start $START --end $END \
    --out data/raw/futures/${SYMBOL}_mark_1h.csv.gz
  
  # Open interest
  python scripts/data/binance_download_open_interest_hist.py \
    --symbol $SYMBOL --period 1h \
    --start $START --end $END \
    --out data/raw/futures/${SYMBOL}_oi.csv.gz
  
  # Basis (may not work for all symbols)
  python scripts/data/binance_download_basis.py \
    --symbol $SYMBOL --period 5m \
    --start $START --end $END \
    --out data/raw/futures/${SYMBOL}_basis.csv.gz || echo "Basis not available for $SYMBOL"
done

# Fetch exchange info
python scripts/data/binance_fetch_futures_exchangeinfo.py \
  --out data/metadata/futures_exchangeInfo.json

# Validate all data
python scripts/data/check_derivatives_sanity.py \
  --symbols BTCUSDT ETHUSDT \
  --start $START --end $END \
  --data-dir data/raw
```

## Rate Limiting

All scripts include:
- Exponential backoff retry logic
- Throttling between requests (default 0.2s, adjustable via `--throttle`)
- Automatic handling of 429 rate limit responses

## Error Handling

Scripts retry on transient failures:
- Network errors: max 6 retries with exponential backoff
- Rate limits: respect `Retry-After` header
- Empty responses: skip forward by one interval to avoid infinite loops

## Data Format

All output files are **gzip-compressed CSV** with:
- Timestamps in **milliseconds since epoch (UTC)**
- Float values formatted to 8 decimal places
- Consistent column naming (see DATA_PROTOCOL.md)

## Requirements

```bash
pip install pandas requests
```

See `requirements.txt` in the repository root.

## Data Protocol

For detailed information about:
- UTC timestamp conventions
- Resampling rules
- File naming conventions
- Expected intervals
- Quality requirements

See [docs/data/DATA_PROTOCOL.md](../../docs/data/DATA_PROTOCOL.md)

## Troubleshooting

### Connection errors
If you get connection errors, check:
1. Internet connectivity
2. Binance API status (https://www.binance.com/en/support/announcement)
3. Rate limits (reduce throttle delay)

### No data returned
Possible causes:
1. Symbol not available for the date range
2. Endpoint doesn't support that symbol (especially for basis)
3. Wrong date format (use YYYY-MM-DD)

### Sanity check failures
Common issues:
1. Incomplete downloads - re-run the download script
2. Network interruptions - check for gaps in timestamps
3. Different date ranges - ensure all downloads use same start/end dates

## Contributing

When adding new data sources:
1. Update DATA_PROTOCOL.md with endpoint documentation
2. Follow existing script structure (retry logic, throttling, error handling)
3. Add appropriate sanity checks
4. Update this README with usage examples
