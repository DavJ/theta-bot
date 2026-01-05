# Spot Bot 2.0 Regime & Strategy Research Note

## Executive summary
- We benchmarked two long/flat intents: EMA-style mean reversion and a Kalman trend/mean filter, both optionally gated by the RegimeEngine (C + psi_logtime).
- On the synthetic 1h sample (cached OHLCV, 240 bars, 0.05% taker fees and 0.5 bps slippage), the Kalman intent delivered small positive returns with negligible drawdowns; mean reversion stayed flat because no z-score entry threshold was met.
- Risk gating leaves Kalman outcomes unchanged on quiet data (no OFF/REDUCE triggers) but provides the discipline to cut exposure when S/C degrade.

## Definitions (aligned with `spot_bot.features.compute_features`)
- Log returns: \(r_t = \log(\mathrm{close}_t / \mathrm{close}_{t-1})\).
- Realized volatility: \( \mathrm{rv}_t = \sqrt{\sum_{i=t-w+1}^{t} r_i^2} \) with window \(w\).
- Log-phase: \( \phi_t = \text{log_phase}(\mathrm{rv}_t, \text{base}=b) \); embeds \(\phi_t\) via \((\cos \phi_t, \sin \phi_t)\).
- Phase concentration: \(C_t = \text{rolling\_phase\_concentration}(\phi, \text{window}=w_c)\).
- Cepstral phase (\(\psi_t\)): complex cepstrum on \(\log(\mathrm{rv})\) using `psi_window`, `cepstrum_domain=logtime`, `[min_bin,max_frac]`.
- Internal concentration: \(C^{\text{int}}_t = \text{rolling\_internal\_concentration}((\cos\phi,\sin\phi), (\cos\psi,\sin\psi), w_c)\).
- Ensemble score: \(S_t = \text{pct\_rank}(C_t)\) averaged with \(C^{\text{int}}\) when available; higher is calmer/cleaner regime.
- RegimeEngine: OFF/REDUCE/ON from thresholds \((s_\text{off}, s_\text{on}, rv_\text{off}, rv_\text{reduce})\) plus smooth `risk_budget`.
- Kalman intent: local linear trend state \([ \text{level}, \text{trend} ]\); innovation z-score \(z_t = (p_t-\hat{\text{level}})/\sqrt{\text{innovation\_var}}\); exposure \(= \sigma(-k z_t)\) clipped to \([0,1]\).

## Experimental setup
- Data: cached OHLCV (`bench_out/ohlcv_BTC_USDT.csv`, `bench_out/ohlcv_ETH_USDT.csv`), 1h bars, 240 samples, no lookahead trimming beyond closed bars.
- Fees/slippage: `--fee-rate 0.0005`, `--slippage-bps 0.5`, max_exposure=1.0, initial_equity=1000 USDT.
- Features: default `FeatureConfig` (base=10, rv_window=24, conc_window=256, psi_window=256, cepstrum_domain=logtime).
- Strategies: mean reversion (EMA/z-score) and Kalman (q_level=1e-4, q_trend=1e-6, r=1e-3, k=1.5, min_bars=10) run as baseline and gated variants.

## Results (from `bench_out/benchmark_strategies_pivot.csv`)

| symbol  | kalman_baseline__final_return | kalman_gated__final_return | meanrev_baseline__final_return | meanrev_gated__final_return | kalman_baseline__max_drawdown | kalman_gated__max_drawdown |
|:-------:|:-----------------------------:|:--------------------------:|:------------------------------:|:---------------------------:|:-----------------------------:|:--------------------------:|
| BTC/USDT | 0.02234 | 0.02234 | 0.00000 | 0.00000 | -0.000108 | -0.000108 |
| ETH/USDT | 0.02234 | 0.02234 | 0.00000 | 0.00000 | -0.000108 | -0.000108 |

Stability scores (median window return − |p10 return| − |median drawdown|) from `bench_out/benchmark_strategies.csv` are ~0.0019 for Kalman variants and 0 for mean reversion on this quiet sample.

## Interpretation
- We do not predict direction; we size into calmer, mean-reverting regimes. Gating uses S (phase concentration) and C_int to avoid high-volatility or incoherent phases.
- Kalman trend/mean filter offers a smooth, deterministic exposure profile that reacts to deviations from the filtered level without overshooting.
- Risk budgeting (RegimeEngine + compute_target_position) remains the arbiter of actual size; intents only suggest desired_exposure.

## Practical recommendations
1. **Mean reversion + risk gating**: Retain as a simple baseline; useful when psi/C show stable concentration and spreads are tight.
2. **Kalman + risk sizing**: Prefer for smoother exposure in trending/mean states; combine with `S`/`C_int` gating to cap size during noisy phases.

## Limitations and next steps
- Synthetic sample lacks real market microstructure (spreads/liq shocks); rerun on live Binance caches for production readiness.
- No shorting; exposure ∈ [0,1]. Extend to symmetrical intent only after validating slippage/fees.
- Feature windows (psi/conc) may emit NaNs early; live runs should use sufficient history and closed-bar truncation.

## Reproducibility
- Features-only pairs scan:  
```bash
  python -m bench.benchmark_pairs --limit-total 8000 --timeframe 1h --out bench_out/benchmark_pairs.csv
```
- Strategy PnL benchmark (both meanrev + Kalman, with optional windows):  
```bash
  python -m bench.benchmark_strategies --limit-total 8000 --timeframe 1h --out bench_out/benchmark_strategies.csv --pivot-out bench_out/benchmark_strategies_pivot.csv --window-days 30
```
- Single backtest example (supports `--strategy kalman`):  
  ```bash
  python -m spot_bot.run_backtest --csv path/to/ohlcv.csv --strategy kalman --kalman-q-level 1e-4 --kalman-q-trend 1e-6 --kalman-r 1e-3 --kalman-k 1.5 --kalman-min-bars 10
  ```
- Live dry-run/paper example (exports features):  
  ```bash
  python -m spot_bot.run_live --mode dryrun --symbol BTC/USDT --timeframe 1h --limit-total 1000 --csv-out bench_out/live_features.csv --csv-out-mode features
  ```
