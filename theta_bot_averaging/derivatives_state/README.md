# Derivatives State Drift Module

This module computes deterministic directional pressure (drift) derived from derivatives market data, NOT from price history.

## Overview

The drift module analyzes derivatives market state (funding rates, open interest, basis) to compute:
- **mu(t)**: Signed drift proxy indicating directional pressure
- **D(t)**: Determinism magnitude = |mu(t)|
- **active(t)**: Gating signal based on threshold or quantile

## Installation

The module is part of `theta_bot_averaging` package. No additional installation required beyond the main project dependencies.

## Usage

### Command Line Interface

Generate drift series for BTCUSDT and ETHUSDT:

```bash
python scripts/generate_drift_series.py \
    --symbols BTCUSDT ETHUSDT \
    --data-dir data/raw \
    --output-dir data/processed/drift \
    --report
```

### Parameters

- `--symbols`: Trading pair symbols to process (default: BTCUSDT ETHUSDT)
- `--data-dir`: Base data directory containing raw derivatives data (default: data/raw)
- `--output-dir`: Output directory for drift files (default: data/processed/drift)
- `--window`: Rolling window for z-score normalization (default: 7D)
- `--quantile`: Quantile threshold for gating (default: 0.85)
- `--threshold`: Optional fixed threshold for gating
- `--alpha`: Weight for overcrowding unwind term (default: 1.0)
- `--beta`: Weight for basis-pressure term (default: 1.0)
- `--gamma`: Weight for expiry/roll pressure term (default: 0.0)
- `--report`: Generate markdown report of top-20 timestamps by D(t)

## Data Requirements

The module requires the following data files following the DATA_PROTOCOL specification:

```
data/raw/
├── spot/
│   └── {SYMBOL}_1h.csv.gz          # Spot klines (OHLCV)
└── futures/
    ├── {SYMBOL}_funding.csv.gz     # Funding rate history (8h intervals)
    ├── {SYMBOL}_oi.csv.gz          # Open interest history
    └── {SYMBOL}_basis.csv.gz       # Basis (mark - spot) or computed fallback
```

Use the mock data generator for testing:

```bash
python scripts/data/generate_mock_derivatives_data.py \
    --symbols BTCUSDT ETHUSDT \
    --start 2024-01-01 --end 2024-10-01 \
    --data-dir data/raw
```

## Output Format

### Drift CSV Files

Output files are saved to `data/processed/drift/{SYMBOL}_1h.csv.gz` with columns:

| Column | Description |
|--------|-------------|
| timestamp | Unix timestamp in milliseconds |
| mu | Total drift = mu1 + mu2 + mu3 |
| D | Determinism magnitude = abs(mu) |
| active | Gating indicator (1 if D > threshold, 0 otherwise) |
| mu1 | Overcrowding unwind component |
| mu2 | Basis-pressure component |
| mu3 | Expiry/roll pressure component (optional) |
| z_funding | Z-scored funding rate |
| z_oi_change | Z-scored open interest change |
| z_basis | Z-scored basis |

### Report Files

When `--report` is specified, markdown reports are generated showing:
- Top 20 timestamps by determinism D(t)
- Context: funding, OI change, basis z-scores
- Summary statistics (mean, median, max, 85th percentile)

## Drift Computation

The drift is computed from standardized derivatives features:

```
z(x) = (x - rolling_mean_7d) / rolling_std_7d

mu1(t) = -alpha * z(OI'(t)) * z(f(t))    # overcrowding unwind
mu2(t) =  beta  * z(OI'(t)) * z(b(t))    # basis-pressure
mu3(t) =  gamma * rho(t) * z(b(t))       # expiry/roll pressure (optional)

mu(t) = mu1 + mu2 + mu3
D(t) = |mu(t)|
```

### Sign Convention

**mu1 (overcrowding unwind)**: Uses negative sign (-alpha) because:
- Positive OI change + positive funding → crowded long position
- Expectation: Market pressure for unwinding → downward drift
- Hence: -alpha to make mu1 negative in this scenario

## Gating

The `active` signal indicates when drift is significant:

```python
# Quantile-based gating
active(t) = D(t) > quantile_85(D)

# Fixed threshold gating  
active(t) = D(t) > threshold

# Combined (OR logic)
active(t) = D(t) > quantile_85(D) OR D(t) > threshold
```

## Module Structure

```
theta_bot_averaging/derivatives_state/
├── __init__.py         # Public API exports
├── loaders.py          # Load spot/futures data per DATA_PROTOCOL
├── features.py         # Z-score normalization, OI change computation
├── drift.py            # Compute mu(t) and D(t)
├── gating.py           # Threshold/quantile gating logic
└── report.py           # Markdown report generation
```

## Python API

```python
from theta_bot_averaging.derivatives_state import (
    load_spot_series,
    load_funding_series,
    load_oi_series,
    load_basis_series,
    compute_zscore,
    compute_oi_change,
    compute_drift,
    compute_determinism,
    apply_combined_gate,
)

# Load data
spot_df = load_spot_series("BTCUSDT", "data/raw")
funding_df = load_funding_series("BTCUSDT", "data/raw")
oi_df = load_oi_series("BTCUSDT", "data/raw")
basis_df = load_basis_series("BTCUSDT", "data/raw")

# Compute features
oi_change = compute_oi_change(oi_df["sumOpenInterest"])
z_funding = compute_zscore(funding_df["fundingRate"], window="7D")
z_oi_change = compute_zscore(oi_change, window="7D")
z_basis = compute_zscore(basis_df["basis"], window="7D")

# Compute drift
mu, mu1, mu2, mu3 = compute_drift(
    z_oi_change=z_oi_change,
    z_funding=z_funding,
    z_basis=z_basis,
    alpha=1.0,
    beta=1.0,
    gamma=0.0,
)

# Compute determinism and gating
D = compute_determinism(mu)
active = apply_combined_gate(D, quantile=0.85, threshold=2.0)
```

## Testing

Run unit tests:

```bash
pytest tests/test_derivatives_state.py -v
```

## References

- [DATA_PROTOCOL.md](../../docs/data/DATA_PROTOCOL.md) - Data loading conventions
- [check_derivatives_sanity.py](../../scripts/data/check_derivatives_sanity.py) - Data validation
- [generate_mock_derivatives_data.py](../../scripts/data/generate_mock_derivatives_data.py) - Mock data generation
