# Spot Bot Regime & Strategy Research Note

## Executive summary
- We evaluate two spot-only intents (mean reversion, Kalman mean/trend) with regime gating from \(C\) and \(\psi\) (Mellin cepstral phase).
- Benchmarks run across BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT, AVAX/USDT with psi modes: none, mellin_cepstrum, mellin_complex_cepstrum.
- Outputs: unified PnL table, stability windows, reproducible CLI commands.

## Problem
Size spot exposure only when regimes are calm/structured; avoid leverage/shorts. Compare base signals and risk gating.

## Data
- OHLCV from Binance via ccxt; 1h bars; `--limit-total` controls history (default 8000).
- Cached per symbol in `bench_out/ohlcv_{symbol}.csv`.

## Feature definitions
- Returns: \( r_t = \log(p_t/p_{t-1}) \).
- Realized vol (rv): \( \mathrm{rv}_t = \sqrt{\sum_{i=t-w+1}^{t} r_i^2} \).
- Log-phase: \( \phi_t = 2\pi \,\mathrm{frac}(\log_b x_t) \) with \(x_t=\mathrm{rv}_t\).
- Concentration: \( C_t = \left|\frac{1}{n}\sum_{i=1}^n e^{j\phi_i}\right| \).
- Mellin cepstral phase: resample \(x[k]\) to log-index \(u=\log k\), compute Mellin spectrum \(X_M = \mathcal{F}_u\{x(e^u)\}\); cepstrum \(c = \mathrm{IFFT}(\log |X_M|)\) (or complex cepstrum using \(\log X_M\)). \(\psi\) is angle of a selected bin (or circular mean over a band).
- Torus embedding: \((\cos\phi, \sin\phi, \cos\psi, \sin\psi)\).
- Internal concentration: \( C^{\text{int}} = \left\|\frac{1}{n}\sum e^{j\phi_i} \oplus e^{j\psi_i}\right\| \) (normalized resultant in \( \mathbb{R}^4 \)).
- Ensemble score: \( S = \text{pct\_rank}(C) \) averaged with \(C^{\text{int}}\) when available.

### Why FFT appears in Mellin approximation
Discrete Mellin is approximated by resampling the signal onto an evenly spaced log-index grid and then applying an FFT. The log-scale change of variables turns Mellin into a Fourier transform in \(u=\log k\), so the FFT is the fast implementation of that step.

## Signal construction
- **Mean reversion**: \( z_t = (p_t-\text{EMA}_t)/\sigma_t \); exposure \(e_t = \text{clip}(-z_t / z_{\max}, [-E,E])\).
- **Kalman**: local linear trend on \(\log p_t\), state \([ \text{level}, \text{slope} ]\). Mean-reversion mode uses residual \(p_t - \hat{\text{level}}\); trend mode uses slope. Exposure scaled by innovation std and clipped to \([-E, E]\).
- **Risk gating**: RegimeEngine on \(C, \psi, C^{\text{int}}, S\). OFF → \(e_t=0\); REDUCE → \(e_t = e_t \cdot \text{risk\_budget}\); ON → unchanged.

## Evaluation protocol
- Backtest is bar-close to next bar-close with fees and slippage on exposure changes.
- Fees: `--fee-rate` (default 0.001). Slippage: `--slippage-bps` (default 1).
- Metrics: final_return_pct, CAGR (annualized for 1h), max_drawdown, realized_vol (annualized), Sharpe (rf=0), trades_count, turnover, time_in_market, average_exposure, fee_paid_estimate.
- Stability: best/worst 30d rolling windows (1h bars → 30*24 bars).

## Results
Run the benchmark (below) to produce `bench_out/strategies.csv` and per-run window JSONs. Pivot/plots are optional.

## Failure modes
- Insufficient history (min bars for Kalman/EMA); returns exposure 0.
- Missing ccxt/theta_features → download not available; use cached OHLCV.
- NaNs in features early windows; gating may default to ON/1.0 where features missing.

## Limitations
- Spot-only, no borrow/short; exposures clipped to [-max_exposure, max_exposure] but execution assumes availability of size.
- Simple fee/slippage model; no order book depth.
- Regime thresholds default; tune per market if needed.

## Next steps
- Add transaction-cost-aware sizing and borrow constraints.
- Experiment with adaptive Kalman noise and psi window lengths.
- Add cross-asset diversification overlay.

## Reproducibility
- Install deps: `pip install -r requirements.txt`
- Dryrun live (features only):  
  `python -m spot_bot.run_live --mode dryrun --symbol BTC/USDT --timeframe 1h --limit-total 1000 --csv-out bench_out/live_features.csv --csv-out-mode features --strategy meanrev`
- Strategy benchmark (PnL, psi modes, windows):  
  `python bench/benchmark_strategies.py --limit-total 8000 --timeframe 1h --out bench_out/strategies.csv --plots-dir bench_out/plots`
- Single backtest example (Kalman):  
  `python -m spot_bot.run_backtest --csv path/to/ohlcv.csv --strategy kalman --kalman-mode meanrev --kalman-q-level 1e-4 --kalman-q-trend 1e-6 --kalman-r 1e-3`
- Features-only pairs scan:  
  `python -m bench.benchmark_pairs --limit-total 8000 --timeframe 1h --out bench_out/benchmark_pairs.csv`
