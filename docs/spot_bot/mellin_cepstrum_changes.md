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
