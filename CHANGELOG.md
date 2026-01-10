# Changelog

## [2026-01-10] Fix Hysteresis Mode Parameter Plumbing in run_live.py

### Fixed
- **hyst_mode Parameter Forwarding**: Fixed missing `hyst_mode` parameter in `compute_step` → `plan_from_live_inputs` call in run_live.py (line 343)
  - Issue: The `--hyst-mode` CLI argument was being accepted but not forwarded to the core engine in certain execution paths
  - Impact: Now `--hyst-mode` CLI argument properly flows through to the core engine in all execution modes (paper, live, dryrun)
  - Related functions already correctly forwarded the parameter (replay, backtest, main loop)

### Added
- **CLI Integration Tests**: Added `TestHystModeCLI` class to `tests/test_hysteresis_fixes.py`
  - `test_hyst_mode_exposure_cli_integration`: Verifies --hyst-mode exposure works end-to-end via CLI
  - `test_hyst_mode_zscore_cli_raises_error_without_zscore`: Verifies proper error when zscore mode used without zscore

### Verified
- All 14 hysteresis tests pass
- All 27 core engine tests pass  
- All 19 run_live tests pass
- Manual acceptance test confirms parameters work:
  - `--hyst-floor 0.02`: 354 trades, turnover=17.60
  - `--hyst-floor 0.12`: 0 trades, turnover=0.00
- All diagnostics present in outputs (target_exposure_raw, target_exposure_final, delta_e, delta_e_min, suppressed, clamped_long_only)
- Fee breakdown present in summary (fees_paid_total, gross_pnl, net_pnl)

### Note
This fix completes the hysteresis implementation. All other requirements (boundary condition fix, hyst_mode implementation, diagnostics) were already implemented in previous changes.

## [2026-01-10] Fix Z-score Hysteresis Mode Validation

### Fixed
- **Z-score Mode Validation**: Fixed `hyst_mode=zscore` to raise a clear RuntimeError when z-score is not available in strategy diagnostics, preventing silent failures.
  - Error message: "hyst_mode=zscore not supported: missing zscore in step context"
  - Previously defaulted to 0.0, making zscore mode ineffective

### Added
- **Test Coverage**: Added `test_zscore_mode_requires_zscore_in_engine` to verify error is raised when zscore mode is used without zscore diagnostics

### Verified
- All existing hysteresis tests pass (12/12)
- Manual acceptance test confirms hysteresis parameters affect trade count:
  - `hyst-floor 0.02`: 134 trades
  - `hyst-floor 0.12`: 0 trades
- Core engine tests pass (27/27)
- Parameter plumbing verified end-to-end

## [2025-01-10] Fix Hysteresis Parameter Plumbing and Diagnostics

### Fixed
- **Boundary Condition**: Changed hysteresis comparison from `delta_e < delta_e_min` to `delta_e <= delta_e_min` for correct floor semantics. This ensures that exposure changes exactly equal to the threshold are suppressed, making `hyst_floor` a true minimum threshold.

### Added
- **hyst_mode Parameter**: Added `hyst_mode` field to `EngineParams` (default: "exposure")
  - Supports two modes: "exposure" and "zscore"
  - "exposure" mode: compares exposure deltas (existing behavior)
  - "zscore" mode: compares z-score deltas when available
  - Invalid modes raise clear error messages
- **Enhanced Diagnostics**: Added hysteresis diagnostic fields to all execution paths:
  - `target_exposure_raw`: Target before long-only clamp
  - `target_exposure_clamped`: Target after long-only clamp, before hysteresis
  - `target_exposure_final`: Final target after hysteresis
  - `delta_e`: Absolute exposure delta
  - `delta_e_min`: Hysteresis threshold used
  - `suppressed`: Boolean flag indicating if hysteresis suppressed the trade
  - `clamped_long_only`: Boolean flag indicating if long-only restriction clamped the target
- **Console Logging**: Updated run_live.py console output to include hysteresis diagnostics
- **Backtest Outputs**: Diagnostic fields now exported to equity and trades CSV files
- **Acceptance Tests**: Added comprehensive test suite in `tests/test_hysteresis_fixes.py`:
  - Parameter plumbing tests (hyst_k, hyst_floor affect delta_e_min)
  - Boundary condition tests (delta_e == delta_e_min suppresses)
  - hyst_mode functionality tests (exposure/zscore modes)
  - Trade reduction tests (higher hyst_floor reduces trades)

### Changed
- **Parameter Forwarding**: `hyst_mode` now properly forwarded through all execution paths:
  - CLI → run_live.py → legacy_adapter → core engine
  - backtest mode → fast_backtest.py → core engine
  - replay mode → legacy_adapter → core engine

### Technical Details
- Modified files:
  - `spot_bot/core/engine.py`: Added hyst_mode to EngineParams, enhanced diagnostics
  - `spot_bot/core/hysteresis.py`: Implemented hyst_mode, fixed boundary condition
  - `spot_bot/core/legacy_adapter.py`: Forward hyst_mode parameter
  - `spot_bot/run_live.py`: Forward hyst_mode, enhanced console logging
  - `spot_bot/backtest/fast_backtest.py`: Forward hyst_mode, export diagnostics
  - `tests/test_hysteresis_fixes.py`: New comprehensive test suite

## [2025-10-25] Atlas Evaluation Document (PR #8)  

- Added `theta_bot_averaging/paper/atlas_evaluation.tex`, a LaTeX document that evaluates the correctness and physical consistency of the Jacobi theta functions (\theta_1 \u2013 \theta_4) used in theta-bot and explains the choice of the nome q and imaginary time.  
- Deprecated any earlier draft evaluation notes in favor of the new comprehensive paper. 
