# Copilot Instructions — Log-Phase / Torus Algorithm (Binance BTC/USDT 1h)

## Goal
Implement an exploratory **log-phase** (fractional-logarithm) feature inspired by a dream:
compare numbers by the *fractional part* of their value or logarithm, interpret that as a **phase** on a circle (S¹),
define **circular distance** on that phase, and optionally extend to a **torus** (T²) for pairs.

This is intended primarily as a **risk/regime feature** (and model feature), not guaranteed alpha.

## Environment / Dependencies
Python 3.10+
Packages: `ccxt`, `pandas`, `numpy`, `matplotlib`, `scipy`
Do **not** use seaborn; use plain matplotlib.

## Data
- Exchange: Binance via `ccxt`
- Symbol: `BTC/USDT`
- Timeframe: `1h`
- Fetch ~3000–6000 OHLCV candles with pagination (use `since`, avoid duplicates, sort by timestamp).
- Use `close` and compute close-to-close returns:
  r_t = close_t / close_{t-1}

## Core math
### Fractional part
frac(x) = x - floor(x)

### Log-phase
For positive x:
phi(x) = frac( log(x) / log(base) ) in [0,1)

Use `base=10.0` by default (also allow `base=np.e` for experiments).
Clamp x with eps to avoid log(0): x = max(x, eps).

Compute:
phi_t = phi(r_t)

### Circular distance (S¹)
Because 0 ≡ 1, use:
circ_dist(a,b) = min(|a-b|, 1-|a-b|)

### Unit-circle embedding (recommended features)
Instead of raw phi, export:
cos_phi = cos(2*pi*phi)
sin_phi = sin(2*pi*phi)

### Rolling phase concentration (clustering / regime)
Let z_t = exp(i*2*pi*phi_t).
For rolling window N (default 256 hours), compute:
C_t = | mean(z_{t-N+1..t}) |

Interpretation:
- ~0  => phase ~ uniform (noise-like)
- high => clustered phase (structure/regime)

Implement `rolling_phase_concentration(phi, window)` efficiently (rolling mean of cos and sin is fine).

## Diagnostics
Run `scipy.stats.kstest(phi, "uniform")` as a quick sanity check (phi is circular, but KS is a fast screen).
Print KS statistic and p-value.
Also print concentration median and 95th percentile.

## Backtest (simple regime filter)
Create two equity curves:
1) Buy&Hold: eq_bh = cumprod(r_t)
2) Filtered exposure:
   - choose threshold thr (default 0.20)
   - weight w_t = 1 if C_t <= thr else 0
   - eq_filtered = cumprod( 1 + w_t*(r_t - 1) )

Report:
- final return (BH vs filtered)
- max drawdown (BH vs filtered)
- time in market (mean(w))

## Plots (matplotlib)
1) Histogram of phi (e.g., 60 bins)
2) Time-series of C_t (phase concentration)
3) Equity curves: buy&hold vs filtered
Allow `--save-plots` to save PNG instead of showing.

## File/Code requirements
- Provide/keep a single runnable script `btc_log_phase.py` with clear functions:
  fetch_ohlcv_binance, frac, log_phase, circ_dist, phase_embedding,
  rolling_phase_concentration, uniformity_test, max_drawdown, risk_filter_backtest, main.
- No hardcoded API keys (public OHLCV only).
- Add concise docstrings explaining the log-phase idea and circular topology.

## Optional extension: torus (T²)
For pairs (e.g., BTC and ETH), define:
Phi_t = (phi_BTC_t, phi_ETH_t) in T²,
and distance on T² using circular distances per component.
(Do not implement unless asked; keep core BTC/USDT 1h working first.)
