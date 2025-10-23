
# Theta Biquaternion Basis — Patch

This patch adds a **theory-first** implementation of the theta basis:
- build finite **q-series** components for Jacobi theta families (θ0–θ3),
- **weighted QR** (or Gram–Schmidt) orthogonalization on a rolling window,
- **projection** onto that orthonormal basis and **direct extrapolation**
  to produce forecasts.

## Files

- `tests_backtest/lib/theta_biquat_basis.py`
  Core math: basis generation, weighted QR, ridge projection, rolling forecast.

- `tests_backtest/cli/run_theta_biquat_basis_csv.py`
  Minimal CLI runner that reads an OHLCV CSV (must contain a `close` column)
  and produces a forecast CSV compatible with your evaluation scripts
  (two variants are emitted: `raw` and `thetaBiquatBasis`).

## Usage

1) Prepare an input CSV with a `close` column (and optionally a `time` column).

2) Run:
```bash
python -m tests_backtest.cli.run_theta_biquat_basis_csv \
  --csv path/to/ohlc.csv \
  --horizons 1h \
  --window 256 \
  --sigma 0.8 --N-even 6 --N-odd 6 \
  --ema-alpha 0.0 --ridge 1e-3 \
  --outdir reports_forecast
```

3) Evaluate with your existing evaluators, e.g.:
```bash
python -m tests_backtest.cli.eval_theta_forecast \
  --csv reports_forecast/forecast_ohlc_win256_h4_biqbasis.csv \
  --metric MSE --outdir reports_forecast

python -m tests_backtest.cli.eval_theta_directional \
  --csv reports_forecast/forecast_ohlc_win256_h4_biqbasis.csv \
  --outdir reports_forecast --fee-bps 0.5
```

> Note: The CLI in this patch reads data from a CSV. It’s intentionally minimal
> to avoid coupling with your live data loaders. If you want, I can wire this
> into your main `run_theta_*` pipeline as another `variant` with the same
> argument style as the rest of the repo.

## Theory defaults

- `tau = i*sigma`, `q = exp(-pi*sigma)`
- truncate even/odd series at `N_even`, `N_odd` such that `q^{n^2}` is small,
- weighted inner product uses EMA row weights and theta q-weights for columns,
- projection uses small ridge (default `1e-3`) for stability.
