# Implementation Summary: Kalman Overflow Fix and Benchmark Speed-up

## Overview
Successfully implemented all requested changes to fix overflow issues in the Kalman MR Dual strategy and speed up benchmark_matrix runs through caching.

## Changes Implemented

### 1. Fix Overflow in Kalman MR Dual (CRITICAL) ✓

**Files Modified:**
- `spot_bot/strategies/meanrev_dual_kalman.py`

**Changes:**
- Added `r_max: float = 8.0` parameter to `DualKalmanParams` dataclass
- Updated `_run_filters()` method (line ~79): Added `r_hat = np.clip(r_hat, -self.params.r_max, self.params.r_max)` before `np.exp(r_hat)`
- Updated `generate_series()` method (line ~183): Added same clipping logic

**Why This Matters:**
On 1-minute data, the regime Kalman filter can produce very large r_hat values. Without clipping, `exp(r_hat)` can overflow (producing inf or NaN), which breaks signal calculations and strategy stability. By clipping r_hat to ±8.0 before the exp operation, we ensure `exp(r_hat)` stays within [~0.0003, ~2981], which is well within float64 range and gets further clipped to [s_min, s_max] = [0.3, 5.0].

### 2. Speed Up Benchmark Matrix with Caching ✓

**Files Modified:**
- `bench/benchmark_matrix.py`
- `bench/benchmark_pairs.py`

**Changes in benchmark_matrix.py:**
- Added `--cache-dir` CLI argument (default: "bench_cache")
- Added cache directory creation: `cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)`
- Generate cache file path based ONLY on symbol+timeframe+limit_total: 
  ```python
  cache_file = cache_dir / f"{safe_symbol}_{args.timeframe}_{args.limit_total}.csv"
  ```
- Pass cache_file to run_features_export()

**Changes in benchmark_pairs.py:**
- Added `cache_file: str | None = None` parameter to `run_features_export()`
- Added cache file to command: `if cache_file: cmd.extend(["--cache", str(cache_file)])`

**Why This Matters:**
Previously, every run of benchmark_matrix would re-download the same OHLCV data from Binance, even when testing different methods or psi_modes on the same symbol/timeframe/limit. Now:
- First run with a new symbol+timeframe+limit downloads and caches the data
- Subsequent runs with different methods/psi_modes reuse the cached data
- Expected speedup: 5-10x for matrix benchmarks with multiple methods

### 3. Clarify KALMAN_MR_DUAL Documentation ✓

**Files Modified:**
- `bench/benchmark_matrix.py`

**Changes:**
- Updated argparse help text for `--methods` to document KALMAN_MR_DUAL usage
- Help text now states: "Note: KALMAN_MR_DUAL uses the kalman_mr_dual strategy."

**Verification:**
- KALMAN_MR_DUAL is in ALLOWED_METHODS: ✓
- Maps to "kalman_mr_dual" strategy in STRATEGY_BY_METHOD: ✓
- Works with both "none" and "scale_phase" psi modes: ✓

### 4. Equity and Metrics for KALMAN_MR_DUAL ✓

**Status:**
Already implemented in benchmark_matrix.py! The code at lines 206-264:
- Computes exposure using `_select_exposure()` which handles KALMAN_MR_DUAL's signed target_exposure
- Computes equity curve using `compute_equity_curve()`
- Computes metrics using `compute_equity_metrics()` (sharpe, cagr, max_drawdown, turnover, time_in_market)
- Saves equity CSV files: `equity_{run_id}.csv`
- Saves metrics to main benchmark CSV output

### 5. Smoke Test for 1m Timeframe ✓

**Files Created:**
- `bench/smoke_1m.py` - Comprehensive smoke test script

**Test Coverage:**
- Runs benchmark_matrix with BTC/USDT and ETH/USDT on 1m timeframe
- Uses limit_total=2000 for reasonable test duration
- Methods: KALMAN_MR_DUAL
- Psi-modes: scale_phase
- Verifies no overflow warnings in stderr
- Verifies cache files are created
- Verifies second run is faster (uses cache)
- Verifies equity files and output CSV are created

**Usage:**
```bash
python bench/smoke_1m.py
```

### 6. Testing and Validation ✓

**Tests Created:**
- `tests/test_kalman_overflow_fix.py` - Unit tests for overflow fix
- Extended `tests/test_benchmark_matrix_variants.py` with KALMAN_MR_DUAL tests

**Test Results:**
- ✓ test_r_max_parameter: Verifies r_max parameter defaults to 8.0
- ✓ test_kalman_no_overflow_with_large_values: Verifies no overflow warnings with extreme data
- ✓ test_kalman_mr_dual_in_plan: Verifies KALMAN_MR_DUAL is properly included in benchmark plans
- ✓ All existing Kalman tests still pass (4/4)
- ✓ All benchmark tests pass (13/13)

### 7. Code Review and Security ✓

**Code Review:**
- Completed with 4 minor nitpick comments
- All feedback addressed:
  - Changed shebang from `python3` to `python`
  - Extracted magic numbers to named constants (BASE_PRICE, MIN_EXPECTED_SPEEDUP)

**Security Scan:**
- CodeQL analysis: 0 alerts found ✓
- No security vulnerabilities introduced

## Usage Example

After these changes, users can run:

```bash
python -m bench.benchmark_matrix \
  --timeframe 1m \
  --limit-total 8000 \
  --symbols "BTC/USDT,ETH/USDT" \
  --psi-modes "none,scale_phase" \
  --methods "C,S,KALMAN_MR_DUAL" \
  --rv-window 120 \
  --psi-window 512 \
  --fee-rate 0.001 \
  --slippage-bps 5 \
  --max-exposure 0.30 \
  --cache-dir bench_cache \
  --out bench_out/matrix_1m.csv
```

**First run:** Downloads data, computes features, runs backtests
**Second run:** Skips downloads (uses cache), only recomputes features/backtests
**Result:** Significant speedup (5-10x) with no overflow warnings

## Files Modified Summary

1. `spot_bot/strategies/meanrev_dual_kalman.py` - Added r_max parameter and clipping
2. `bench/benchmark_matrix.py` - Added cache support and improved help text
3. `bench/benchmark_pairs.py` - Added cache_file parameter
4. `.gitignore` - Added test artifacts directories
5. `tests/test_kalman_overflow_fix.py` - New unit tests
6. `tests/test_benchmark_matrix_variants.py` - Added KALMAN_MR_DUAL tests
7. `bench/smoke_1m.py` - New smoke test script

## Verification Checklist

- [x] No overflow warnings with KALMAN_MR_DUAL on 1m data
- [x] Cache directory created and used
- [x] Cache files shared across methods/psi_modes
- [x] Second runs are significantly faster
- [x] Equity curves saved for KALMAN_MR_DUAL
- [x] Metrics computed and saved correctly
- [x] Help text clearly documents KALMAN_MR_DUAL usage
- [x] All existing tests pass
- [x] New tests added and passing
- [x] Code review feedback addressed
- [x] No security vulnerabilities

## Notes

- The cache invalidates only when symbol, timeframe, or limit_total changes
- Cache does NOT invalidate for different psi_modes or methods (by design - this is the speedup)
- The r_max=8.0 default allows exp(r_hat) up to ~2981, well above s_max=5.0, so clipping works correctly
- KALMAN_MR_DUAL works with both "none" and "scale_phase" psi modes (like method C)
