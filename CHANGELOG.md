# Changelog  

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
