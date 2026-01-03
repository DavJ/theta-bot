# Derivatives Drift Implementation Summary

## Overview

Successfully implemented a new module for computing deterministic directional pressure (drift) from derivatives market data, NOT from price history.

## Deliverables

### 1. New Package: `theta_bot_averaging/derivatives_state/`

Created complete package with the following modules:

- **`loaders.py`** (210 lines)
  - Load spot klines, funding rates, open interest, and basis data
  - Follow DATA_PROTOCOL conventions (1h grid, UTC timestamps, forward-fill)
  - Handle both direct basis data and fallback computation (mark - spot)

- **`features.py`** (129 lines)
  - Compute z-score normalization: `z(x) = (x - rolling_mean) / rolling_std`
  - Compute OI change: `OI'(t) = diff(log(OI))`
  - Align multiple time series to common index
  - Rolling window default: 7 days

- **`drift.py`** (92 lines)
  - Compute drift components:
    - `mu1(t) = -alpha * z(OI'(t)) * z(f(t))` (overcrowding unwind)
    - `mu2(t) = beta * z(OI'(t)) * z(b(t))` (basis-pressure)
    - `mu3(t) = gamma * rho(t) * z(b(t))` (expiry/roll pressure, optional)
  - Total drift: `mu(t) = mu1 + mu2 + mu3`
  - Determinism: `D(t) = |mu(t)|`

- **`gating.py`** (102 lines)
  - Quantile-based gating: `active(t) = D(t) > quantile_85`
  - Fixed threshold gating: `active(t) = D(t) > threshold`
  - Combined gating: OR logic of both methods

- **`report.py`** (90 lines)
  - Generate markdown reports of top-N timestamps by D(t)
  - Include context: z_funding, z_oi_change, z_basis
  - Summary statistics: mean, median, max, 85th percentile

- **`__init__.py`** (44 lines)
  - Public API exports for all key functions

- **`README.md`** (194 lines)
  - Comprehensive documentation
  - Usage examples (CLI and Python API)
  - Formula explanations
  - Data requirements

### 2. CLI Tool: `scripts/generate_drift_series.py`

**Features:**
- Process multiple symbols (default: BTCUSDT, ETHUSDT)
- Configurable parameters:
  - Rolling window for z-score (default: 7D)
  - Quantile threshold (default: 0.85)
  - Optional fixed threshold
  - Drift component weights (alpha, beta, gamma)
- Optional markdown report generation
- Outputs CSV files with 10 columns:
  - timestamp, mu, D, active, mu1, mu2, mu3, z_funding, z_oi_change, z_basis

**Example Usage:**
```bash
python scripts/generate_drift_series.py \
    --symbols BTCUSDT ETHUSDT \
    --window 7D \
    --quantile 0.85 \
    --alpha 1.0 --beta 1.0 --gamma 0.0 \
    --report
```

### 3. Testing: `tests/test_derivatives_state.py`

Created 8 comprehensive unit tests:
1. `test_compute_zscore` - Validates z-score normalization
2. `test_compute_oi_change` - Validates OI log-difference
3. `test_compute_drift` - Validates drift formula components
4. `test_compute_determinism` - Validates D(t) = |mu(t)|
5. `test_apply_quantile_gate` - Validates quantile gating
6. `test_apply_threshold_gate` - Validates fixed threshold gating
7. `test_apply_combined_gate` - Validates OR logic
8. `test_drift_with_custom_weights` - Validates custom weight parameters

**Test Results:**
- All 8 new tests passing ✓
- Full test suite: 57 tests passing ✓
- No test regressions

### 4. Validation

Generated drift series for BTCUSDT and ETHUSDT using mock data:

**BTCUSDT Results:**
- 6,545 records generated
- Mean D(t): 1.0083
- Median D(t): 0.6867
- Max D(t): 6.7769
- 85th percentile D(t): 2.0556

**ETHUSDT Results:**
- 6,545 records generated
- Mean D(t): 1.0167
- Median D(t): 0.6749
- Max D(t): 6.2787
- 85th percentile D(t): 2.0683

**Sample Top Timestamp (BTCUSDT, highest D(t)):**
- Date: 2024-04-05 17:00
- D(t): 6.7769
- mu(t): -6.7769
- z_funding: +2.2397
- z_oi_change: +1.8094
- z_basis: -1.5057
- Interpretation: High positive funding + rising OI + negative basis → strong negative drift (expect downward pressure)

## Output Format

CSV files saved to `data/processed/drift/{SYMBOL}_1h.csv.gz`:

```csv
timestamp,mu,D,active,mu1,mu2,mu3,z_funding,z_oi_change,z_basis
1704153600000,0.160607,0.160607,0,0.289907,-0.129300,0.0,0.879027,-0.329804,0.392051
1704157200000,-2.982713,2.982713,1,-1.295092,-1.687621,0.0,0.849597,1.524360,-1.107102
...
```

## Code Quality

### Code Review
Addressed all 5 review comments:
1. ✓ Optimized timestamp conversion (removed redundant operations)
2. ✓ Improved mu3 allocation efficiency (named Series)
3. ✓ Replaced magic number with constant `MS_TO_NS`
4. ✓ Changed epsilon handling (1e-10 instead of NaN)
5. ✓ Fixed test assertion logic (>= instead of >)

### Security Scan
- CodeQL scan completed: **0 alerts found** ✓
- No security vulnerabilities detected

### Linting
- No new linting issues introduced
- Follows existing code style conventions
- Proper docstrings for all functions

## Mathematical Foundation

### Sign Convention Explained

**mu1 (overcrowding unwind)**: Uses negative sign (-alpha)
- When OI increases AND funding rate is positive → crowded long position
- Market expectation: Pressure for unwinding → downward price movement
- Therefore: mu1 should be negative in this scenario
- Formula: `mu1 = -alpha * z(OI') * z(f)` ensures this sign convention

**mu2 (basis-pressure)**: Uses positive sign (+beta)
- When OI increases AND basis widens → increased demand for futures
- Market expectation: Futures premium indicates upward pressure
- Therefore: mu2 should be positive when both are rising
- Formula: `mu2 = +beta * z(OI') * z(b)` captures this relationship

### Gating Rationale

Only use drift signal when determinism is high:
- `active(t) = True` when `D(t) > quantile_85` OR `D(t) > fixed_threshold`
- Filters out low-confidence periods
- Focuses on strong directional pressure events
- Typical activation: ~15% of time periods

## Data Protocol Compliance

All loaders follow DATA_PROTOCOL.md specifications:
- ✓ UTC-only timestamps
- ✓ Canonical close-time grid
- ✓ 1h baseline interval
- ✓ Funding rate forward-fill (8h → 1h)
- ✓ OI/basis resampling (last observation per hour)
- ✓ File naming conventions
- ✓ CSV format with gzip compression

## Acceptance Criteria Met

✓ Running CLI produces drift series for BTCUSDT and ETHUSDT
✓ Outputs CSV files with all required columns
✓ Produces reports listing top-20 timestamps by D(t)
✓ Reports include context (funding, OI change, basis)
✓ All formulas implemented correctly
✓ Gating logic works (quantile and threshold)
✓ Comprehensive testing coverage
✓ No security vulnerabilities
✓ Clean code review

## Future Enhancements (Optional)

1. **Expiry/roll pressure (mu3)**: Requires delivery futures metadata
   - Load expiry dates from `futures_exchangeInfo.json`
   - Compute `rho(t) = days_to_expiry / max_days`
   - Enable with `--gamma > 0`

2. **Real-time computation**: Adapt for live data streams
   - Incremental z-score updates
   - Streaming computation

3. **Multi-symbol aggregation**: Cross-asset drift analysis
   - Compute market-wide drift
   - Detect regime shifts

4. **Backtesting integration**: Use drift signals in trading strategies
   - Combine with existing theta_bot models
   - Position sizing based on D(t)

## Files Changed

```
theta_bot_averaging/derivatives_state/
  __init__.py                  (44 lines, new)
  loaders.py                   (210 lines, new)
  features.py                  (129 lines, new)
  drift.py                     (92 lines, new)
  gating.py                    (102 lines, new)
  report.py                    (90 lines, new)
  README.md                    (194 lines, new)

scripts/
  generate_drift_series.py     (264 lines, new)

tests/
  test_derivatives_state.py    (217 lines, new)
```

**Total:** 9 files, 1,342 lines of new code

## Conclusion

Successfully implemented a complete derivatives drift module that:
- Computes directional pressure from market microstructure
- Provides clean, well-tested API
- Includes production-ready CLI tool
- Follows project conventions and data protocol
- Passes all quality and security checks

The module is ready for use in analyzing derivatives market dynamics and can be integrated into trading strategies that leverage directional pressure signals.
