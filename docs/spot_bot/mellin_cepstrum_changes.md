# Mellin Cepstrum Implementation Changes

## Overview

This document describes the implementation of Mellin-transform-based cepstrum as an alternative to existing FFT-based cepstrum analysis in the theta-bot spot trading system.

## Files Modified

### 1. `theta_features/cepstrum.py`

**New Functions:**
- `mellin_transform()`: Computes Mellin transform via log-domain resampling
- `mellin_cepstrum()`: Computes real Mellin cepstrum (log magnitude only)
- `mellin_complex_cepstrum()`: Computes complex Mellin cepstrum with true phase
- `mellin_cepstral_phase()`: Extracts psi from real Mellin cepstrum
- `mellin_complex_cepstral_phase()`: Extracts psi from complex Mellin cepstrum
- `rolling_mellin_cepstral_phase()`: Rolling window wrapper for real Mellin
- `rolling_mellin_complex_cepstral_phase()`: Rolling window wrapper for complex Mellin
- `_extract_psi_from_cepstrum()`: Helper for extracting phase with configurable aggregation

**Key Features:**
- Log-domain resampling for numerical Mellin transform computation
- Optional exponential weighting with sigma parameter
- Phase unwrapping and optional detrending for complex cepstrum
- Configurable phase aggregation (peak or circular mean)
- Consistent interface with existing FFT-based functions

### 2. `spot_bot/features/feature_pipeline.py`

**Changes to `FeatureConfig`:**
Added new parameters:
- `mellin_grid_n`: Grid size for Mellin transform (default: 256)
- `mellin_sigma`: Exponential weighting parameter (default: 0.0)
- `mellin_eps`: Epsilon for log stability (default: 1e-12)
- `mellin_detrend_phase`: Phase detrending flag (default: True)
- `psi_min_bin`: Minimum bin for psi extraction (default: 2)
- `psi_max_frac`: Maximum fraction for band (default: 0.25)
- `psi_phase_agg`: Aggregation method - "peak" or "cmean" (default: "peak")
- `psi_phase_power`: Power for circular mean weighting (default: 1.0)

**Updated Functions:**
- `_compute_psi_with_debug()`: Now routes to Mellin functions when `psi_mode` is "mellin_cepstrum" or "mellin_complex_cepstrum"
- Added imports for new Mellin functions

### 3. `spot_bot/run_live.py`

**CLI Argument Changes:**
- Extended `--psi-mode` choices to include:
  - `mellin_cepstrum`: Real Mellin cepstrum
  - `mellin_complex_cepstrum`: Complex Mellin cepstrum with true phase

**New CLI Arguments:**
- `--mellin-grid-n`: Grid size for Mellin transform
- `--mellin-sigma`: Sigma parameter for exponential weighting
- `--mellin-eps`: Epsilon for log stability
- `--mellin-detrend-phase`: Enable phase detrending
- `--psi-min-bin`: Minimum bin for psi extraction
- `--psi-max-frac`: Maximum fraction for psi band
- `--psi-phase-agg`: Phase aggregation method (peak or cmean)
- `--psi-phase-power`: Power for circular mean weighting

**FeatureConfig Construction:**
Updated to pass all new Mellin parameters from CLI arguments.

### 4. Tests

**New Test File:** `tests/test_mellin_cepstrum.py`
- Tests basic Mellin transform computation
- Tests real and complex Mellin cepstral phase
- Tests rolling window wrappers
- Tests edge cases (empty arrays, NaN inputs, short series)
- Tests debug output functionality
- Tests different aggregation methods

## Backward Compatibility

All existing functionality remains intact:
- Default `psi_mode` is still "cepstrum" (FFT-based)
- Existing FFT-based modes ("cepstrum", "complex_cepstrum") work unchanged
- All existing tests pass without modification
- FeatureConfig defaults maintain backward compatibility

## Usage Examples

### Using Mellin Cepstrum in CLI

```bash
# Real Mellin cepstrum
python spot_bot/run_live.py \
  --psi-mode mellin_cepstrum \
  --mellin-grid-n 256 \
  --mellin-sigma 0.0 \
  --psi-phase-agg peak

# Complex Mellin cepstrum with phase detrending
python spot_bot/run_live.py \
  --psi-mode mellin_complex_cepstrum \
  --mellin-grid-n 512 \
  --mellin-sigma 0.5 \
  --mellin-detrend-phase \
  --psi-phase-agg cmean \
  --psi-phase-power 1.5
```

### Using Mellin Cepstrum in Code

```python
from spot_bot.features import FeatureConfig, compute_features

# Configure for Mellin complex cepstrum
cfg = FeatureConfig(
    psi_mode="mellin_complex_cepstrum",
    mellin_grid_n=256,
    mellin_sigma=0.0,
    mellin_detrend_phase=True,
    psi_min_bin=2,
    psi_max_frac=0.25,
    psi_phase_agg="peak",
)

features = compute_features(ohlcv_df, cfg)
```

## Parameter Tuning Guidelines

### Grid Size (`mellin_grid_n`)
- Larger values (512-1024) provide higher frequency resolution
- Smaller values (64-128) are faster but less precise
- Default 256 provides good balance for most use cases

### Sigma (`mellin_sigma`)
- 0.0 (default): No exponential weighting
- Positive values: Emphasize higher frequencies
- Negative values: Emphasize lower frequencies
- Typical range: -1.0 to 1.0

### Phase Detrending (`mellin_detrend_phase`)
- True (default): Removes linear trend from unwrapped phase
- False: Uses raw unwrapped phase
- Detrending often improves stability

### Phase Aggregation (`psi_phase_agg`)
- "peak" (default): Uses the bin with maximum magnitude
- "cmean": Circular mean of all bins in band, weighted by magnitude
- Circular mean can be more robust but slower

### Phase Power (`psi_phase_power`)
- Only used with "cmean" aggregation
- Higher values give more weight to stronger components
- Typical range: 0.5 to 2.0
- Default 1.0 uses linear weighting

## Performance Considerations

- Mellin transform requires log-domain resampling and FFT
- Computational cost similar to FFT-based cepstrum
- Grid size has linear impact on computation time
- Rolling windows process one window at a time (no optimization yet)

## Known Limitations

- No scipy dependency: Uses numpy-only implementation
- Phase detrending uses simple linear regression
- No multi-scale or hierarchical Mellin transforms yet
- Rolling implementations are not vectorized

## Tuning

### Overview

The `spot_bot/tune_mellin.py` script provides a reproducible parameter sweep and tuning framework for selecting optimal Mellin cepstrum parameters. It supports three tuning modes:

1. **Regime mode**: Tunes parameters for `mellin_cepstrum` to optimize regime stability
2. **Phase mode**: Tunes parameters for `mellin_complex_cepstrum` to optimize phase timing
3. **Both mode**: Runs regime sweep first, then uses best configs as base for phase sweep

### Usage

Basic usage for regime parameter tuning:

```bash
python spot_bot/tune_mellin.py \
  --csv path/to/ohlcv.csv \
  --mode regime \
  --top 5 \
  --out-csv regime_results.csv
```

Tune phase parameters using a specific base configuration:

```bash
python spot_bot/tune_mellin.py \
  --csv path/to/ohlcv.csv \
  --mode phase \
  --top 10 \
  --out-csv phase_results.csv
```

Run comprehensive sweep with walk-forward validation:

```bash
python spot_bot/tune_mellin.py \
  --csv path/to/ohlcv.csv \
  --mode both \
  --walk-forward \
  --train-bars 1000 \
  --test-bars 500 \
  --slippage-bps 2.0 \
  --fee-rate 0.001 \
  --max-exposure 0.5 \
  --out-csv full_results.csv
```

### Command-Line Arguments

**Required:**
- `--csv PATH`: Path to OHLCV CSV file with columns: timestamp, open, high, low, close, volume

**Backtest Parameters:**
- `--slippage-bps FLOAT`: Slippage in basis points (default: 0.0)
- `--fee-rate FLOAT`: Transaction fee rate (default: 0.0005)
- `--max-exposure FLOAT`: Maximum exposure fraction (default: 1.0)
- `--initial-equity FLOAT`: Initial equity amount (default: 1000.0)

**Output Parameters:**
- `--out-csv PATH`: Path to save results CSV (optional)
- `--top N`: Number of top configurations to print (default: 5)

**Tuning Mode:**
- `--mode {regime,phase,both}`: Tuning mode (default: regime)

**Walk-Forward Validation:**
- `--walk-forward`: Enable walk-forward validation
- `--train-bars N`: Number of training bars (default: 1000)
- `--test-bars N`: Number of test bars (default: 500)

**Reproducibility:**
- `--seed INT`: Random seed for reproducibility (default: 42)

### Parameter Grids

**Regime Mode** (`mellin_cepstrum`):
- `psi_window`: [128, 256, 512]
- `mellin_sigma`: [-0.5, 0.0, 0.5]
- `mellin_grid_n`: [128, 256, 512]
- `psi_min_bin`: [2, 4, 6]
- `psi_max_frac`: [0.15, 0.2, 0.25, 0.3]

Fixed: `psi_phase_agg="peak"`

**Phase Mode** (`mellin_complex_cepstrum`):
- `mellin_detrend_phase`: [True, False]
- `psi_phase_agg`: ["peak", "cmean"]
- `psi_phase_power`: [0.5, 1.0, 1.5, 2.0]
- `mellin_eps`: [1e-14, 1e-12, 1e-10]

Uses base configuration from regime mode or default values.

### Interpreting Results

The script outputs configurations ranked by:
1. **Sharpe ratio** (descending): Higher is better - measures risk-adjusted returns
2. **Max drawdown** (descending): Less negative is better - measures downside risk
3. **Turnover** (ascending): Lower is better - reduces transaction costs

**Key Metrics:**
- `sharpe`: Annualized Sharpe ratio (assumes hourly bars, 24*365 periods/year)
- `final_return`: Total return over the backtest period
- `max_drawdown`: Maximum peak-to-trough decline
- `volatility`: Annualized volatility of returns
- `turnover`: Trading activity relative to initial equity
- `trades`: Number of trades executed
- `time_in_market`: Fraction of time with non-zero position

**Walk-Forward Metrics** (when enabled):
- `mean_sharpe`: Average Sharpe ratio across test folds
- `mean_max_drawdown`: Average max drawdown across test folds
- `num_folds`: Number of walk-forward folds evaluated

### Output CSV Format

The results CSV contains:
- All configuration parameters (e.g., `psi_window`, `mellin_sigma`, etc.)
- All performance metrics (e.g., `sharpe`, `max_drawdown`, `final_return`, etc.)
- One row per configuration tested
- Sorted by ranking criteria

### Best Practices

1. **Start with regime mode**: Find stable regime detection parameters first
2. **Use walk-forward validation**: More robust than single backtest
3. **Check multiple metrics**: Don't optimize for Sharpe alone - consider drawdown and turnover
4. **Validate on held-out data**: Test best configs on separate time periods
5. **Consider transaction costs**: Use realistic `--fee-rate` and `--slippage-bps`
6. **Beware of overfitting**: More parameters → higher risk of curve-fitting

### Example Output

```
================================================================================
TOP 5 CONFIGURATIONS
================================================================================

Rank 1:
  Metrics:
    sharpe              :     1.2345
    final_return        :     0.3456
    max_drawdown        :    -0.0789
    volatility          :     0.2345
    turnover            :     2.3456
    trades              :    45.0000
  Configuration:
    psi_mode            : mellin_cepstrum
    psi_window          :        256
    mellin_grid_n       :        256
    mellin_sigma        :        0.0
    psi_min_bin         :          4
    psi_max_frac        :       0.25
    psi_phase_agg       :       peak

...
```

### Tips for Parameter Selection

**psi_window**: Larger windows capture longer-term patterns but are slower to adapt
- Use 128 for fast-changing markets
- Use 256 for balanced performance
- Use 512 for stable, slow-moving markets

**mellin_sigma**: Controls frequency weighting
- Negative values emphasize low frequencies (long-term trends)
- Zero (default) provides balanced weighting
- Positive values emphasize high frequencies (short-term patterns)

**mellin_grid_n**: Higher values provide better frequency resolution but are slower
- Use 128 for speed
- Use 256 for balanced performance (recommended)
- Use 512 for maximum precision

**psi_min_bin** / **psi_max_frac**: Control the frequency band for phase extraction
- Smaller min_bin includes lower frequencies
- Larger max_frac includes higher frequencies
- Typical range captures medium-frequency cycles

**mellin_detrend_phase**: Removes linear trends from phase
- True (default) often improves stability
- False preserves raw phase information

**psi_phase_agg**: How to aggregate phase from multiple frequency bins
- "peak" uses the strongest frequency component (faster, simpler)
- "cmean" uses circular mean weighted by magnitude (more robust, slower)

**psi_phase_power**: Only affects "cmean" aggregation
- Higher values give more weight to dominant components
- Lower values spread weight more evenly
