# Data Protocol

This document defines the canonical, reproducible layout for real-market datasets used by the derivatives SDE pipeline.

## Core Principles
- **UTC everywhere**: every timestamp is in UTC milliseconds.
- **Canonical index**: the primary time index is the **close-time** on a 1h grid.
- **Deterministic output**: given the same symbol, interval, and date window, scripts produce identical gzip CSV files.
- **No API keys / no evasion**: only public Binance endpoints or the official bulk archive are used.

## Directory Layout & File Names
```
data/
├── raw/
│   ├── spot/
│   │   └── {SYMBOL}_1h.csv.gz
│   └── futures/
│       ├── {SYMBOL}_funding.csv.gz
│       ├── {SYMBOL}_oi.csv.gz
│       ├── {SYMBOL}_mark_1h.csv.gz
│       └── {SYMBOL}_basis.csv.gz
├── processed/
│   └── derivatives_sde/
│       └── {SYMBOL}_1h.csv.gz
└── metadata/
    └── futures_exchangeInfo.json
```

## Required Datasets & Columns

### Spot klines (1h)
- Columns: `open_time_ms, close_time_ms, open, high, low, close, volume`
- Index: `close_time_ms` (canonical 1h close)

### Funding (native 8h)
- Columns: `timestamp_ms, funding_rate`
- Resample to 1h by **forward-filling** each published rate across the next hours until the next funding point.

### Open Interest history
- Columns: `timestamp_ms, open_interest` (numeric). If available, also store `open_interest_value`.
- Resample to 1h:
  - If higher frequency: downsample using the **last** value in the hour.
  - If lower frequency: forward-fill and track a mask column `open_interest_is_filled` when carried forward.

### Mark price klines (1h)
- Columns: `open_time_ms, close_time_ms, open, high, low, close, volume` (volume optional).
- Index: `close_time_ms` (canonical 1h close).

### Basis (derived)
- Definition: `basis(t) = log(mark_close(t)) - log(spot_close(t))`
- Stored in `data/raw/futures/{SYMBOL}_basis.csv.gz`
- Columns: `timestamp_ms, basis, mark_close, spot_close`

### Processed derivatives SDE output
- File: `data/processed/derivatives_sde/{SYMBOL}_1h.csv.gz`
- Columns include inputs (spot_close, funding_rate, open_interest, basis, mark_close), derived series (returns, doi, z-scores), and outputs (`mu`, `sigma`, `lambda`, `active`).

## Resampling Rules (to 1h close grid)
- Funding: forward-fill 8h points to each hour until the next publication.
- Open Interest: `last()` within each 1h window; forward-fill if native cadence is lower than 1h with mask.
- Mark/Spot klines: already 1h; if higher frequency, aggregate to OHLCV by hour close.
- Basis: computed directly on the aligned 1h close grid.

## Data Quality & Sanity Expectations
1. **Monotonic UTC DatetimeIndex** (no duplicates).
2. **Expected step**: 1h for spot/mark/basis/open-interest; 8h for raw funding before resample.
3. **No NaNs** in required columns after resampling (except where a mask indicates forward-fill).
4. **Overlap window**: report intersection of spot, funding, oi, mark, basis.
5. **Value checks**:
   - Funding typically in [-0.01, 0.01]
   - Open interest strictly positive
   - Mark close near spot close (order-of-magnitude consistency)

## HTTP & Caching Policy
- Use a single `requests.Session()` per script.
- Pace requests to **≤ 2 req/s** (configurable).
- Exponential backoff on 429 / 418 / 5xx.
- **Caching**: if the target output file exists and `--force` is not set, the download is skipped.
- Fallback: when API limits block open-interest downloads, use official bulk archives from `https://data.binance.vision`.

## Naming Examples
- `data/raw/spot/BTCUSDT_1h.csv.gz`
- `data/raw/futures/ETHUSDT_funding.csv.gz`
- `data/raw/futures/BTCUSDT_oi.csv.gz`
- `data/raw/futures/ETHUSDT_mark_1h.csv.gz`
- `data/raw/futures/BTCUSDT_basis.csv.gz`
- `data/processed/derivatives_sde/ETHUSDT_1h.csv.gz`

## Usage Examples
```bash
# Metadata
python scripts/data/binance_fetch_metadata.py

# Spot klines
python scripts/data/binance_download_spot_klines.py \
  --symbols BTCUSDT,ETHUSDT --interval 1h \
  --start 2024-01-01 --end 2024-10-01

# Funding
python scripts/data/binance_download_funding.py \
  --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-10-01

# Mark price
python scripts/data/binance_download_mark_klines.py \
  --symbols BTCUSDT,ETHUSDT --interval 1h \
  --start 2024-01-01 --end 2024-10-01

# Open interest with bulk fallback
python scripts/data/binance_download_open_interest_hist.py \
  --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-10-01

# Basis (derived)
python scripts/data/build_basis.py --symbols BTCUSDT,ETHUSDT --interval 1h

# Sanity checks
python scripts/data/check_derivatives_sanity.py \
  --symbols BTCUSDT,ETHUSDT --start 2024-01-01 --end 2024-10-01
```

## Notes
- All times are UTC close-times on a 1h grid unless stated otherwise.
- No API keys required; be polite to the endpoints.
- Official bulk archive (`data.binance.vision`) is the only fallback when the live API is constrained.
