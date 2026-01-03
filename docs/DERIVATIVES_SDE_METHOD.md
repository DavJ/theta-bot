# Derivatives SDE decomposition

This module implements an SDE-first split between deterministic drift and stochastic diffusion for spot log returns:

```
ds(t) = mu(d(t), t) dt + sigma(t) dW(t)
```

Where:
* `s(t)` is log spot price
* `d(t)` is a derivatives state vector (funding, OI, basis, expiry proximity)
* `mu(d,t)` is a drift proxy (directional pressure)
* `sigma(t)` is diffusion intensity estimated from trailing volatility
* `Lambda(t) = |mu(t)| / sigma(t)` is a determinism score

## Data inputs
All inputs follow the existing data protocol and are loaded from `data/raw`:
* Spot klines: `data/raw/spot/{SYMBOL}_1h.csv.gz`
* Funding: `data/raw/futures/{SYMBOL}_funding.csv.gz` (forward-filled to 1h)
* Open interest: `data/raw/futures/{SYMBOL}_oi.csv.gz`
* Basis: `data/raw/futures/{SYMBOL}_basis.csv.gz` when available

Series are aligned on the intersection of their UTC 1h timestamps.

## State construction
`theta_bot_averaging.derivatives_sde.state.build_state` computes:
* `r`: log spot returns
* `oi_change`: log difference of open interest
* Standardized z-scores for funding, OI change, and basis using a trailing rolling window (default 168h) with clipping.
* Masks for missing inputs.

## Drift model
`theta_bot_averaging.derivatives_sde.mu_model.compute_mu_components` implements:

```
mu1 = -alpha * z(oi_change) * z(funding)
mu2 =  beta * z(oi_change) * z(basis)
mu3 =  gamma * rho(t) * z(basis)   # rho optional
mu  = mu1 + mu2 + mu3
```

Defaults: `alpha=1.0`, `beta=0.5`, `gamma=0.5`.

## Diffusion model
`theta_bot_averaging.derivatives_sde.sigma_model.compute_sigma` estimates `sigma(t)` via trailing rolling volatility (default 168h) with an optional EWMA blend and an epsilon floor.

## Decomposition and gating
`theta_bot_averaging.derivatives_sde.sde_decompose.decompose_symbol` computes `mu`, `sigma`, `Lambda`, and an activity gate:
* Fixed threshold `tau`
* Quantile threshold `tau_quantile`

Outputs are written to `data/processed/derivatives_sde/{SYMBOL}_1h.csv.gz`.

## Fitting
`theta_bot_averaging.derivatives_sde.fit.walk_forward_fit` regresses returns on the base mu components with expanding-window folds to estimate `alpha`, `beta`, and `gamma`.

## Evaluation
`theta_bot_averaging.derivatives_sde.eval.evaluate_bias` measures conditional bias for horizons `{1,3,6,12,24}` on the active set:
* Sign agreement between `mu` and future returns
* Conditional means up/down
* Effect size and Lambda-decile monotonicity
* Controls for inactive periods and shuffled mu

## Reports and scripts
* Decomposition report: `reports/DERIVATIVES_SDE_DECOMPOSITION.md`
* Evaluation report: `reports/DERIVATIVES_SDE_EVAL.md`

CLI entry points:
```
python -m theta_bot_averaging.derivatives_sde.run_decompose --symbols BTCUSDT --tau_quantile 0.85
python -m theta_bot_averaging.derivatives_sde.run_eval --symbols BTCUSDT,ETHUSDT --tau_quantile 0.85
```
