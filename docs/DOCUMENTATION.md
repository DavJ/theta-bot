# Theta-Bot: Comprehensive Documentation

> **Version**: Spot Bot 2.0  
> **Purpose**: Predictive algorithmic trading engine for crypto spot markets  
> **Theoretical Basis**: Complex Consciousness Theory (CCT) / Unified Biquaternion Theory (UBT) + Jacobi theta functions

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Descriptions](#2-component-descriptions)
3. [Mathematical and Statistical Foundations](#3-mathematical-and-statistical-foundations)
4. [Installation and Configuration](#4-installation-and-configuration)
5. [Backtesting, Optimization, and Live Trading](#5-backtesting-optimization-and-live-trading)
6. [API Reference](#6-api-reference)
7. [FAQ and Common Issues](#7-faq-and-common-issues)
8. [Contribution Guide](#8-contribution-guide)

---

## 1. Architecture Overview

Theta-Bot is a modular, multi-layer trading engine that converts raw OHLCV market data into trading signals via feature engineering, regime detection, and pluggable strategies. The system is designed for **long/flat spot trading** (no shorts, no leverage) on crypto markets.

### High-Level Data Flow

```
Exchange / CSV Data (OHLCV)
        │
        ▼
┌─────────────────────────────┐
│  Feature Extraction         │  theta_features/
│  RV, φ, ψ, C, S            │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  Regime Engine              │  spot_bot/regime/
│  ON / REDUCE / OFF          │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  Strategy Layer             │  spot_bot/strategies/
│  intent (desired_exposure)  │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  Core Engine                │  spot_bot/core/
│  Hysteresis → TradePlan     │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  Execution                  │  spot_bot/run_live.py
│  DRYRUN / PAPER / LIVE      │
└──────────────┬──────────────┘
               │
        ▼
┌─────────────────────────────┐
│  State Persistence          │  spot_bot/persist/
│  SQLite (bars, trades, eq.) │
└─────────────────────────────┘
```

### Package Overview

| Package / Module | Purpose |
|-----------------|---------|
| `spot_bot/` | Production trading system (Spot Bot 2.0) |
| `theta_features/` | Log-phase & scale-phase feature extraction |
| `theta_bot_averaging/` | Advanced ML walk-forward validation framework |
| `theta_basis_4d.py` | 4D Jacobi theta basis generation |
| `theta_transform.py` | Theta basis projection |
| `theta_predictor.py` | Walk-forward v9 predictor (biquaternion drift) |
| `bench/` | Benchmark harness for comparing strategies |
| `tools/` | Evaluation scripts (eval_metrics.py) |
| `tests/` | Comprehensive pytest test suite (~324 tests) |

---

## 2. Component Descriptions

### 2.1 Feature Pipeline (`spot_bot/features/`, `theta_features/`)

The feature pipeline converts raw OHLCV data into structured regime features.

**`compute_features(df, cfg) → DataFrame`** (`spot_bot/features/feature_pipeline.py`)

Input: DataFrame with columns `[timestamp, open, high, low, close, volume]`

Output columns:

| Column | Description | Range |
|--------|-------------|-------|
| `rv` | Realized volatility (24h rolling) | > 0 |
| `phi` | Log-phase: `frac(log(RV) / log(base))` | [0, 1) |
| `cos_phi`, `sin_phi` | Phase embedding on unit circle | [-1, 1] |
| `psi` | Scale-phase: `frac(log(RV/median) / log(base))` | [0, 1) |
| `psi_mode` | PSI computation mode label | string |
| `C` | Phase concentration \|E[e^{i2πφ}]\| | [0, 1] |
| `C_int` | Internal torus concentration on (φ, ψ) | [0, 1] |
| `S` | Ensemble score combining C, C_int | ≥ 0 |

**Configuration (`FeatureConfig`):**

```python
from spot_bot.features import FeatureConfig

cfg = FeatureConfig(
    base=1.1,           # Log base for phase computation
    rv_window=24,       # Rolling window for RV (bars)
    conc_window=24,     # Rolling window for concentration
    psi_mode="scale_phase",   # Phase mode: "scale_phase" or "none"
    psi_window=256,     # Window for median RV in scale-phase
)
```

---

### 2.2 Regime Engine (`spot_bot/regime/regime_engine.py`)

The regime engine classifies market state based on feature values.

**States:**

| State | Condition | Action |
|-------|-----------|--------|
| `ON` | `S ≥ s_on` and `rv < rv_off` | Full strategy exposure |
| `REDUCE` | `s_off < S < s_on` | Scale exposure by budget |
| `OFF` | `S < s_off` or `rv > rv_off` | Zero exposure |

**`RegimeEngine` parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `s_off` | -0.1 | Score threshold → OFF state |
| `s_on` | 0.0 | Score threshold → ON state |
| `s_budget_low` | -0.1 | Budget floor (maps to 0) |
| `s_budget_high` | 1.0 | Budget ceiling (maps to 1) |
| `rv_off` | None | Max RV before forcing OFF |
| `rv_reduce` | None | RV threshold for REDUCE state |
| `rv_guard` | None | RV-based budget suppression |

---

### 2.3 Strategies (`spot_bot/strategies/`)

All strategies implement `generate_intent(features_df) → Intent` where `Intent` has:
- `desired_exposure` ∈ [0, 1]: fraction of capital to invest
- `reason`: human-readable label for the signal

#### Mean Reversion (`mean_reversion.py`)

Identifies deviations from the exponential moving average and trades the reversion.

```
z_score = (price - EMA(price, span)) / rolling_std(price, window)
signal = -z_score  # long when below EMA

Exposure:
  if signal < entry_z:  exposure = 0
  else: exposure = lerp(min_exp, max_exp, (signal - entry_z) / (full_z - entry_z))
```

**Key parameters:** `ema_span=20`, `std_window=30`, `entry_z=0.5`, `full_z=2.0`

#### Kalman Filter (`kalman.py`)

Local linear trend model. Tracks `[level, trend]` state.

```
State transition:   x_{t+1} = [[1,1],[0,1]] @ x_t + w_t
Measurement model:  y_t = [1,0] @ x_t + v_t

Exposure: sigmoid(k * (level - price) / sqrt(innovation_var))
  → 0.5 when price == level
  → approaches 1 when price far below level (oversold)
  → approaches 0 when price far above level (overbought)
```

**Key parameters:** `q_level=1e-4`, `q_trend=1e-6`, `r=1e-3`, `k=1.5`

#### Dual Kalman – Mean Reversion (`meanrev_dual_kalman.py`)

Hybrid strategy combining:
1. Mean-reversion z-score gating (entry/exit timing)
2. Dual Kalman filters (level + trend) for exposure scaling

Best used when the market alternates between trending and mean-reverting regimes.

#### LSTM-Kalman (`lstm_kalman.py`)

Combines an LSTM sequence encoder with Kalman-filtered state estimate. Requires PyTorch.

#### Dual-Stream Model (`theta_bot_averaging/models/dual_stream.py`)

Advanced ML strategy. Two parallel feature streams fused into a return prediction:

```
Input OHLCV window (theta_window bars)
    ├── Theta Stream: theta basis projection → GRU → theta_features
    └── Mellin Stream: Mellin transform → MLP → mellin_features
            │
            ▼
    Gated Fusion → predicted_return
            │
            ▼
    Signal: sign(predicted_return) if |predicted_return| > threshold else 0
```

---

### 2.4 Core Engine (`spot_bot/core/`)

The core engine is the single source of truth for all trading math. It ensures consistent behavior across all execution modes.

#### Hysteresis (`hysteresis.py`)

Prevents excessive position churn by requiring a minimum change in exposure before acting.

```
delta_e_min = max(
    hyst_floor,
    hyst_k * fee_rate * (1 + slippage_bps/10000 + spread_bps/20000)
)

if |target_exposure - current_exposure| < delta_e_min:
    # Suppress the action – not worth the fees
    pass
```

**Parameters:**
- `hyst_k`: Multiplier (e.g., 5.0 means: don't trade unless PnL covers 5× fees)
- `hyst_floor`: Minimum threshold regardless of fees (e.g., 0.02)
- `hyst_mode`: `"exposure"` (default) or `"zscore"` (requires strategy zscore output)

#### Trade Planner (`trade_planner.py`)

Converts a target exposure fraction → actionable trade plan.

```
target_base = target_exposure * total_equity / price
delta_base  = target_base - current_base

if |delta_base * price| < min_notional:
    # Skip – too small to trade
    pass

if step_size:
    delta_base = round(delta_base / step_size) * step_size
```

#### Cost Model (`cost_model.py`)

Computes execution cost for a given trade:

```
cost = |notional| * (fee_rate + slippage_bps/10000 + spread_bps/20000)
```

#### Portfolio (`portfolio.py`)

Tracks capital, position, and equity:

```
equity    = usdt + base * price
exposure  = base * price / equity
realized_pnl = Σ (sell_price - cost_basis) * qty
```

---

### 2.5 Backtest Engine (`spot_bot/backtest/fast_backtest.py`)

**`run_backtest(df, ...) → (equity_df, trades_df, summary)`**

1. Computes features for the entire time series (vectorized, no lookahead)
2. Applies valid_mask (drops NaN rows from warm-up period)
3. Loops bar-by-bar:
   - Generate intent from strategy
   - Apply hysteresis filter
   - Plan trade
   - Simulate order execution (market or limit fill)
   - Update portfolio state

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy_name` | `"meanrev"` | Strategy identifier |
| `fee_rate` | 0.001 | Taker fee (0.1%) |
| `slippage_bps` | 0 | Price slippage in basis points |
| `max_exposure` | 0.3 | Maximum fraction of capital invested |
| `initial_usdt` | 1000 | Starting capital |
| `hyst_k` | 5.0 | Hysteresis multiplier |
| `hyst_floor` | 0.02 | Minimum hysteresis threshold |
| `bar_state` | `"closed"` | Use closed-bar prices |
| `min_notional` | 5.0 | Minimum trade size (USDT) |

---

### 2.6 Walk-Forward Validation (`theta_bot_averaging/validation/walkforward.py`)

**`run_walkforward(config_path) → dict`**

Orchestrates time-series cross-validation with purging and embargo to prevent data leakage.

```
Timeline:
│────────────────────────────────────────────────────────────────│
│  FOLD 1                                                         │
│  [─── TRAIN ────][purge][──── TEST ────][──── embargo ────]    │
│                                                                 │
│  FOLD 2                                                         │
│         [─────────── TRAIN ───────────][purge][── TEST ──][em] │
│                                                                 │
│  ... (n_splits total)                                           │
```

**Outputs (per run):**
- `runs/{timestamp}/{config_name}/metrics.json` – fold + aggregate metrics
- `runs/{timestamp}/{config_name}/predictions.parquet` (or `.csv`) – predictions
- `runs/{timestamp}/{config_name}/backtest.json` – trade-level backtest

---

### 2.7 State Persistence (`spot_bot/persist/`)

All state is stored in a local SQLite database (`--db bot.db`).

**Tables:**

| Table | Contents |
|-------|----------|
| `bars` | Raw OHLCV data per bar |
| `features` | Computed features per bar |
| `intents` | Strategy intents (desired_exposure, reason) |
| `trades` | Executed fills (qty, price, fee, realized_pnl) |
| `equity_snapshots` | Equity curve (usdt, base, equity, exposure) |
| `kv_store` | Key-value pairs (e.g., `last_closed_ts`) |

**Recovery:** On restart, the bot reads `last_closed_ts` from `kv_store` to resume from where it left off without re-processing.

---

### 2.8 Evaluation Metrics (`tools/eval_metrics.py`)

Standalone evaluation script that processes CSV datasets and computes:

| Metric | Description |
|--------|-------------|
| `corr_pred_true` | Pearson correlation: predicted vs actual returns |
| `hit_rate` | Fraction of correct direction predictions (zeros = miss) |
| `total_pnl_usdt` | Total profit/loss in USDT |
| `end_capital_usdt` | Final capital after simulated trading |
| `avg_monthly_pnl_usdt` | Average monthly profit/loss |

---

## 3. Mathematical and Statistical Foundations

### 3.1 Jacobi Theta Functions

The core theoretical framework. The theta function is:

```
Θ₃(z, τ) = Σ_{n=-∞}^{∞}  q^{n²} · e^{2πinz}

where:
  q = e^{iπτ}  (nome, |q| < 1 for convergence)
  z = phase variable (complex)
  τ = complex time parameter (τ = t + iψ)
```

**4D Basis** (`theta_basis_4d.py`): A 4-dimensional orthonormal basis is constructed using:
- 32 discrete Fourier modes: n ∈ [-16, 16]
- Sampled over 8 × 8 × 8 grid (frequency × phase × imaginary-time)
- QR decomposition enforces orthonormality (numerical error < 10⁻¹⁵)
- Verified: Hermitian symmetry (< 10⁻¹⁸), energy conservation

**Application**: Project a price return series onto the theta basis to extract market microstructure coefficients.

---

### 3.2 Complex-Time Representation (CCT/UBT)

The system models market time as:

```
τ = t + iψ

where:
  t  = real (clock) time
  ψ  = imaginary part (hidden phase / regime variable)
```

This captures two aspects of market dynamics:
- **Real time**: fundamental timeline for price evolution
- **Imaginary time**: technical/sentiment regime embedded in volatility structure

The scale-phase ψ is computed from realized volatility (see §3.4).

---

### 3.3 Realized Volatility (RV)

```
RV_t = sqrt( (1/W) · Σ_{i=t-W+1}^{t} log(P_i / P_{i-1})² )

W = rv_window (default: 24 bars)
```

More robust than simple return volatility because it uses log-returns and is well-behaved for compounding analysis.

---

### 3.4 Log-Phase φ

Maps realized volatility to a cyclic phase variable:

```
φ_t = {log₁₀(RV_t)}   [fractional part: {x} = x - floor(x)]
φ_t ∈ [0, 1)           (fractional part only)
```

**Interpretation**: Each order of magnitude in RV corresponds to one full cycle. This normalizes the volatility signal across different market regimes.

**Phase embedding** (unit circle coordinates):
```
cos_φ = cos(2π · φ)
sin_φ = sin(2π · φ)
```

---

### 3.5 Scale-Phase ψ

The imaginary-time variable:

```
ψ_t = { log_base( RV_t / median(RV_{t-W:t}) ) }

where {x} = x - floor(x) denotes the fractional part.
```

**Interpretation**: Measures where the current RV sits relative to its recent median, on a log scale. High ψ means unusually volatile relative to recent history.

---

### 3.6 Phase Concentration C

Measures how tightly the phase values cluster on the unit circle:

```
C = | E[ e^{i·2π·φ} ] | = sqrt( mean(cos_φ)² + mean(sin_φ)² )
C ∈ [0, 1]
```

- `C → 1`: all recent φ values cluster near the same point (coherent regime)
- `C → 0`: φ values uniformly distributed (disordered / transitioning)

**Interpretation**: High concentration indicates a stable, predictable volatility regime.

---

### 3.7 Internal Torus Concentration C_int

Extends concentration to the 2-torus (φ, ψ):

```
C_int = || [mean(cos_φ), mean(sin_φ), mean(cos_ψ), mean(sin_ψ)] || / 2
C_int ∈ [0, 1]
```

Captures joint coherence on the (φ, ψ) phase space.

---

### 3.8 Ensemble Score S

Combines C and C_int into a single market quality signal:

```
S = weighted_combination(C, C_int, ...)
```

High S → high regime coherence → strategy is allowed to take positions.
Low S → regime is unstable → strategy reduces or exits exposure.

---

### 3.9 Kalman Filter

The Kalman strategy models price as a locally linear trend:

```
State:         x_t = [level_t, trend_t]ᵀ
Transition:    x_{t+1} = F x_t + w_t
               F = [[1, 1], [0, 1]]     (level + trend model)
Measurement:   y_t = H x_t + v_t
               H = [1, 0]               (observe level only)

Process noise:   Q = diag(q_level, q_trend)
Measurement noise: R = r

Exposure: σ( k · z )  where  z = (level - price) / sqrt(S_t)
  σ(x) = 1 / (1 + e^{-x})  (sigmoid)
  S_t = innovation variance
```

Parameters `q_level` and `q_trend` control how quickly the filter tracks vs. smooths.

---

### 3.10 Mellin Transform (Advanced)

Used in the dual-stream model for scale-invariant frequency features:

```
M[f](s) = ∫₀^∞ f(t) · t^{s-1} dt

Discrete approximation:
  m_k = Σ_t x_t · t^{α + iω_k}   for ω_k = k·Δω

Properties:
  - Scale invariance: M[f(at)](s) = a^{-s} M[f](s)
  - Useful for signals with power-law amplitude distributions
  - Captures multiplicative structure in price dynamics
```

**Parameters**: `mellin_k=16` (frequency samples), `mellin_alpha=0.5` (real exponent), `mellin_omega_max=1.0`

---

### 3.11 Walk-Forward Validation (Preventing Lookahead Bias)

Standard k-fold cross-validation is **invalid** for time series because future data leaks into training. Theta-Bot uses **Purged Time Series Split**:

```
Purge:   Remove training samples whose labels overlap with the test period
Embargo: After each test fold, skip a gap before the next training window

Effect: 100% guarantee that no future information enters training
```

This is critical for financial ML where even small data leaks can produce overfit results with no real edge.

---

## 4. Installation and Configuration

### 4.1 Requirements

- Python 3.10+
- Dependencies: see `requirements.txt`

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
pyyaml>=6.0
scipy>=1.10
pytest>=7.4
pyarrow>=22.0.0
requests>=2.31.0
matplotlib>=3.5
ccxt>=4.0
```

Optional:
- `torch` (PyTorch): required for dual-stream model
- `statsmodels`: for advanced statistical analysis

### 4.2 Installation

```bash
# Clone the repository
git clone https://github.com/DavJ/theta-bot.git
cd theta-bot

# Install dependencies
pip install -r requirements.txt

# Optional: install theta_bot package in development mode (has its own pyproject.toml)
pip install -e theta_bot/
```

### 4.3 Configuration

**Global defaults** (`spot_bot/config.yaml`):
```yaml
timeframe: "1h"
strategy: "meanrev"
fee_rate: 0.001        # Taker fee (0.1%)
slippage_bps: 0.5      # Price slippage (0.5 bps)
max_exposure: 0.30     # Maximum 30% of capital invested
min_notional: 5.0      # Minimum trade size (USDT)
psi_mode: "scale_phase"
psi_window: 256
rv_window: 24
conc_window: 24
hyst_k: 5.0
hyst_floor: 0.02
```

**Walk-forward config** (`configs/btc_1h.yaml`):
```yaml
data_path: "real_data/BTCUSDT_1h.csv"
horizon: 1             # Predict 1-bar ahead
threshold_bps: 10      # Signal threshold in basis points
model_type: "logit"    # or "dual_stream"
signal_mode: "threshold"  # or "quantile"
fee_rate: 0.0004       # Binance BNB discount rate
slippage_bps: 1.0
spread_bps: 0.5
n_splits: 5            # Number of time-series folds
purge: 1               # Purge 1 bar around fold boundaries
embargo: 1             # Skip 1 bar after test set ends
output_dir: "runs"
```

**Dual-stream config** (`configs/dual_stream_example.yaml`):
```yaml
model_type: "dual_stream"
theta_window: 48       # 48-hour context window
theta_q: 0.9           # Theta function decay parameter
theta_terms: 8         # Number of theta basis terms
mellin_k: 16           # Mellin frequency samples
mellin_alpha: 0.5
mellin_omega_max: 1.0
torch_epochs: 50       # Training epochs (requires PyTorch)
torch_batch_size: 32
torch_lr: 0.001
```

### 4.4 Data Download

```bash
# Download OHLCV data from Binance
python download_market_data.py --symbol BTCUSDT --interval 1h --limit 2000

# Validate downloaded data quality
python validate_real_data.py --csv real_data/BTCUSDT_1h.csv

# Alternative: use the theta-bot downloader with multiple pairs
python download_binance_data.py --symbols BTCUSDT ETHUSDT --interval 1h --days 30
```

---

## 5. Backtesting, Optimization, and Live Trading

### 5.1 Dryrun (No Trading – Feature Inspection)

Useful for inspecting feature computation and strategy intent without placing any orders:

```bash
python -m spot_bot.run_live \
  --mode dryrun \
  --symbol BTC/USDT \
  --timeframe 1h \
  --limit-total 1000 \
  --strategy meanrev \
  --csv-out bench_out/live_features.csv \
  --csv-out-mode features
```

---

### 5.2 Backtesting from CSV

#### Quick Backtest (`run_backtest.py`)

```bash
python -m spot_bot.run_backtest \
  --csv data/btc_1h.csv \
  --strategy kalman \
  --kalman-mode meanrev \
  --slippage-bps 1 \
  --save-plots plots/
```

#### Full Backtest (`run_live.py`)

More control over all parameters:

```bash
python -m spot_bot.run_live \
  --mode backtest \
  --csv-in data/btc_1h.csv \
  --timeframe 1h \
  --strategy kalman_mr_dual \
  --psi-mode scale_phase \
  --psi-window 512 \
  --rv-window 24 \
  --fee-rate 0.001 \
  --slippage-bps 5 \
  --max-exposure 0.30 \
  --min-notional 5 \
  --hyst-k 5.0 \
  --hyst-floor 0.02 \
  --out-equity bench_out/equity.csv \
  --out-trades bench_out/trades.csv \
  --out-summary bench_out/summary.json
```

#### Limit Maker Backtest

Simulate limit-maker order execution:

```bash
python -m spot_bot.run_live \
  --mode backtest \
  --csv-in data/btc_1h.csv \
  --strategy meanrev \
  --order-type limit_maker \
  --maker-offset-bps 1 \
  --max-spread-bps 15 \
  --maker-fee-rate 0.0001 \
  --taker-fee-rate 0.001
```

---

### 5.3 Walk-Forward Validation (ML Models)

```bash
# Baseline logistic regression
python -m theta_bot_averaging.validation.walkforward configs/btc_1h.yaml

# Dual-stream model (PyTorch)
python -m theta_bot_averaging.validation.walkforward configs/dual_stream_example.yaml

# Quantile signal mode
python -m theta_bot_averaging.validation.walkforward configs/dual_stream_quantile.yaml
```

**Output structure:**
```
runs/
  {timestamp}/
    {config_name}/
      metrics.json          ← fold + aggregate metrics
      predictions.parquet   ← all fold predictions
      backtest.json         ← simulated trading results
```

---

### 5.4 Hyperparameter Optimization

```bash
# Grid search for optimal parameters
python optimize_hyperparameters.py --csv real_data/BTCUSDT_1h.csv

# Sweep theta basis hyperparameters
python btc_log_phase_sweep.py
```

---

### 5.5 Benchmarking Multiple Strategies

```bash
python -m bench.benchmark_strategies \
  --limit-total 8000 \
  --timeframe 1h \
  --out bench_out/strategies.csv \
  --plots-dir bench_out/plots
```

---

### 5.6 Paper Trading (Simulated Live)

```bash
python -m spot_bot.run_live \
  --mode paper \
  --symbol BTC/USDT \
  --timeframe 1h \
  --db bot.db \
  --initial-usdt 1000 \
  --fee-rate 0.001 \
  --max-exposure 0.3 \
  --strategy meanrev
```

The bot runs in a loop, fetching new bars at each interval and simulating fills at the close price. State is persisted to `bot.db` for restarts.

---

### 5.7 Live Trading (Real Orders)

⚠️ **WARNING**: Live trading uses real money. Thoroughly backtest and paper-trade before enabling.

```bash
# Market orders (taker fees)
python -m spot_bot.run_live \
  --mode live \
  --i-understand-live-risk \
  --symbol BTC/USDT \
  --timeframe 1h \
  --db bot.db \
  --fee-rate 0.001 \
  --max-exposure 0.30 \
  --strategy meanrev

# Limit maker orders (lower fees, potential rebates)
python -m spot_bot.run_live \
  --mode live \
  --i-understand-live-risk \
  --symbol BTC/USDT \
  --timeframe 1h \
  --db bot.db \
  --order-type limit_maker \
  --maker-offset-bps 1 \
  --max-spread-bps 15 \
  --order-validity-seconds 120 \
  --maker-fee-rate 0.0001 \
  --taker-fee-rate 0.001
```

**Required**: `--i-understand-live-risk` flag explicitly acknowledges live trading risk.

---

### 5.8 Evaluation Metrics

```bash
# Evaluate local CSV datasets
python3 tools/eval_metrics.py \
  --repo-root . \
  --start-capital 1000 \
  --taker-fee 0.001 \
  --no-network

# Include Binance API data
python3 tools/eval_metrics.py \
  --repo-root . \
  --start-capital 1000 \
  --taker-fee 0.001 \
  --pairs BTCUSDT ETHUSDT
```

---

## 6. API Reference

### 6.1 `spot_bot.run_live` CLI

```
python -m spot_bot.run_live [OPTIONS]

Core Options:
  --mode {dryrun,paper,backtest,replay,live}
                          Execution mode (default: dryrun)
  --symbol SYMBOL         Trading pair, e.g. BTC/USDT (default: BTC/USDT)
  --timeframe TIMEFRAME   Bar interval (default: 1h)
  --strategy {meanrev,kalman,kalman_mr_dual,lstm_kalman}
                          Trading strategy (default: meanrev)
  --db PATH               SQLite database path (default: bot.db)
  --initial-usdt FLOAT    Starting capital in USDT (default: 1000.0)
  --fee-rate FLOAT        Taker fee rate (default: 0.001)
  --max-exposure FLOAT    Max fraction of capital invested (default: 0.30)
  --min-notional FLOAT    Minimum trade size in USDT (default: 5.0)

Backtest Options:
  --csv-in PATH           Input OHLCV CSV file
  --limit-total INT       Max bars to process (default: 2000)
  --out-equity PATH       Output equity curve CSV
  --out-trades PATH       Output trades CSV
  --out-summary PATH      Output summary JSON

Feature Parameters:
  --psi-mode {scale_phase,none}
                          Scale-phase computation mode (default: scale_phase)
  --psi-window INT        Window for median RV in scale-phase (default: 256)
  --rv-window INT         Realized volatility window (default: 24)
  --conc-window INT       Concentration rolling window (default: 24)

Hysteresis Parameters:
  --hyst-k FLOAT          Hysteresis multiplier (default: 5.0)
  --hyst-floor FLOAT      Minimum hysteresis threshold (default: 0.02)
  --hyst-mode {exposure,zscore}
                          Hysteresis computation mode (default: exposure)

Order Execution:
  --order-type {market,limit_maker}
                          Order type (default: market)
  --maker-offset-bps FLOAT  Limit order offset in bps (default: 1.0)
  --max-spread-bps FLOAT  Maximum spread to allow orders (default: 20.0)
  --order-validity-seconds INT
                          Cancel unfilled orders after N seconds (default: 60)
  --maker-fee-rate FLOAT  Maker fee rate (default: same as --fee-rate)
  --taker-fee-rate FLOAT  Taker fee rate (default: same as --fee-rate)

Safety:
  --i-understand-live-risk  Required for --mode live
```

---

### 6.2 `run_backtest(df, ...)` Python API

```python
from spot_bot.backtest import run_backtest

equity_df, trades_df, summary = run_backtest(
    df=df,                    # DataFrame: timestamp, open, high, low, close, volume
    timeframe="1h",           # Bar interval string
    strategy_name="meanrev",  # Strategy identifier
    psi_mode="scale_phase",   # Feature mode
    psi_window=256,           # Scale-phase window
    rv_window=24,             # RV window
    conc_window=24,           # Concentration window
    base=1.1,                 # Log base for phase computation
    fee_rate=0.001,           # Taker fee
    slippage_bps=0.5,         # Slippage
    spread_bps=0.0,           # Spread cost
    max_exposure=0.3,         # Max exposure fraction
    initial_usdt=1000.0,      # Starting capital
    min_notional=5.0,         # Min trade size
    step_size=None,           # Lot size (None = continuous)
    bar_state="closed",       # Use closed-bar prices
    log=True,                 # Print progress
    hyst_k=5.0,               # Hysteresis multiplier
    hyst_floor=0.02,          # Min hysteresis threshold
    hyst_mode="exposure",     # Hysteresis mode
)

# equity_df: timestamp, equity, usdt, base, exposure
# trades_df: timestamp, action, qty, price, notional, fee, realized_pnl
# summary: dict with trades_count, final_equity, total_pnl, max_drawdown, sharpe
```

---

### 6.3 `compute_features(df, cfg)` Python API

```python
from spot_bot.features import FeatureConfig, compute_features

cfg = FeatureConfig(
    base=1.1,
    rv_window=24,
    conc_window=24,
    psi_mode="scale_phase",
    psi_window=256,
)

features = compute_features(df, cfg)
# Returns DataFrame with: rv, phi, cos_phi, sin_phi, psi, C, C_int, S
```

---

### 6.4 `run_walkforward(config_path)` Python API

```python
from theta_bot_averaging.validation import run_walkforward

result = run_walkforward("configs/btc_1h.yaml")
# result["metrics"]["aggregate"] → {sharpe, hit_rate, pnl, max_drawdown, ...}
# result["output_dir"] → path to run outputs
```

---

### 6.5 `tools/eval_metrics.py` Functions

```python
from tools import eval_metrics

# Compute simple returns from price series
returns = eval_metrics.compute_returns(prices)

# Pearson correlation between predicted and actual returns
corr = eval_metrics.compute_correlation(pred_returns, true_returns)

# Hit rate: fraction of correct direction predictions
hit = eval_metrics.compute_hit_rate(pred_returns, true_returns)

# Full simulation with simple strategy
result = eval_metrics.simulate_trading(df, fee_rate=0.001, start_capital=1000.0)
# result: {total_pnl_usdt, end_capital_usdt, avg_monthly_pnl_usdt, num_trades}

# Evaluate a CSV file end-to-end
metrics = eval_metrics.evaluate_dataset(
    csv_path="data.csv",
    fee_mode="taker_fee",   # or "no_fees"
    taker_fee=0.001,
    start_capital=1000.0,
)
```

---

## 7. FAQ and Common Issues

### Q: How much data do I need for a reliable backtest?

**A:** The warm-up period consumes `max(rv_window, conc_window, psi_window) + feature_lag ≈ 50–300` bars before producing valid features. We recommend at least **1,000 bars** (41+ days at 1h) for a meaningful backtest. For walk-forward validation, aim for **3,000–10,000 bars** per fold.

---

### Q: The backtest shows 0 trades. Why?

**A:** This is usually caused by:
1. **Insufficient data**: Not enough bars after warm-up to produce valid features. Check that `features.shape[0] > 0` after filtering.
2. **Hysteresis threshold too high**: Reduce `--hyst-floor` (default 0.02) or `--hyst-k` (default 5.0).
3. **All regime OFF**: The `S` score may be consistently below `s_off`. Check feature values with `--mode dryrun`.
4. **min_notional too high**: With small capital, reduce `--min-notional`.

---

### Q: I get `ValueError: Insufficient data to run backtest`

**A:** After feature computation and NaN filtering, no valid bars remain. Solutions:
- Use more input data (increase `--limit-total` or provide a longer CSV)
- Reduce `rv_window` and `conc_window` (shorter warm-up)
- Check the input data has non-constant prices (all-same prices → zero RV → NaN features)

---

### Q: `KeyError: 'open'` or `KeyError: 'high'`

**A:** Your input DataFrame is missing OHLC columns. The backtest requires `[open, high, low, close, volume]`. If you're using a CSV with only `close`, add synthetic OHLC columns:

```python
df["open"] = df["close"].shift(1, fill_value=df["close"].iloc[0])
df["high"] = df["close"] * 1.001
df["low"]  = df["close"] * 0.999
df["volume"] = 1000.0
```

---

### Q: `ValueError: Invalid frequency: H`

**A:** Your pandas version is ≥ 2.2, which deprecated uppercase frequency aliases. Replace `freq='H'` with `freq='h'`, `freq='T'` with `freq='min'`, etc. See [pandas migration guide](https://pandas.pydata.org/docs/whatsnew/v2.2.0.html#deprecated-aliases).

---

### Q: Walk-forward predictions CSV: `predicted_return` is the index, not a column

**A:** This was a bug where `to_csv(index=False)` was used, causing the date-time index to be lost and `predicted_return` to become the first column (then parsed as index). Fixed in the current version.

---

### Q: How do I use `--hyst-mode zscore`?

**A:** Z-score hysteresis requires the strategy to output a `zscore` value in its diagnostics. Currently, the `kalman_mr_dual` strategy supports this. If you use `meanrev` or `kalman` with `--hyst-mode zscore`, it will raise a `RuntimeError` because those strategies don't provide a z-score.

---

### Q: What are the expected performance metrics on real data?

**A:** Based on testing:
- **Correlation**: 0.05–0.15 is meaningful (random is ~0)
- **Hit rate**: 52–56% is a tradable edge (random is 50%)
- **Sharpe ratio**: > 0.5 annualized (synthetic data shows higher – real data will be lower)

Do not expect synthetic-data results (Sharpe > 14) to hold on live markets.

---

### Q: How do I restart a stopped bot without reprocessing old data?

**A:** If using `--db bot.db`, the bot automatically reads `last_closed_ts` from the SQLite `kv_store` table and resumes from the last processed bar. Just re-run the same command.

---

### Q: The CI pipeline fails with `ModuleNotFoundError: No module named 'sklearn'`

**A:** Install scikit-learn: `pip install scikit-learn`. It is required for baseline models in `theta_bot_averaging/`.

---

## 8. Contribution Guide

### 8.1 Development Setup

```bash
git clone https://github.com/DavJ/theta-bot.git
cd theta-bot
pip install -r requirements.txt
pip install -e theta_bot/  # optional editable install

# Run tests to verify setup
python -m pytest tests/ -q
```

### 8.2 Code Organization

- **One concern per module**: strategies in `spot_bot/strategies/`, features in `spot_bot/features/`, etc.
- **Single source of truth**: All trading math lives in `spot_bot/core/` (engine, portfolio, cost model, hysteresis)
- **Pluggable strategies**: New strategies implement `generate_intent(features_df) → Intent`
- **No lookahead**: Features and strategies must use only past data at each bar

### 8.3 Adding a New Strategy

1. Create `spot_bot/strategies/my_strategy.py`:

```python
from spot_bot.strategies.base import BaseStrategy, Intent
import pandas as pd


class MyStrategy(BaseStrategy):
    """Brief description of strategy."""

    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param

    def generate_intent(self, features_df: pd.DataFrame) -> Intent:
        """Generate trading intent from feature data.
        
        Args:
            features_df: DataFrame with columns [close, rv, phi, C, S, ...]
        
        Returns:
            Intent with desired_exposure ∈ [0, 1] and reason string.
        """
        # Your logic here
        exposure = 0.5  # example: always 50% invested
        return Intent(desired_exposure=exposure, reason="my_reason")
```

2. Register it in `spot_bot/strategies/__init__.py`

3. Add it to the `--strategy` argument in `spot_bot/run_live.py`

4. Write tests in `tests/test_my_strategy.py`

### 8.4 Writing Tests

- Tests live in `tests/` and follow `pytest` conventions
- Use `pd.date_range(..., freq="h")` (lowercase) for pandas >= 2.2 compatibility
- Provide OHLCV DataFrames with all 5 columns: `open, high, low, close, volume`
- Use `tmp_path` fixture for file I/O tests (avoids hardcoded paths)
- Use synthetic data with enough variability (at least 200–500 bars, oscillating prices)

```python
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(bars=200, seed=42):
    """Create synthetic OHLCV DataFrame for testing."""
    np.random.seed(seed)
    idx = pd.date_range("2024-01-01", periods=bars, freq="h")
    t = np.linspace(0, 4 * np.pi, bars)
    close = 20000 + 500 * np.sin(t) + np.cumsum(np.random.randn(bars) * 5)
    close = close.clip(min=1000)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({
        "timestamp": idx,
        "open": open_,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": 1000.0,
    })


def test_my_strategy():
    df = _make_ohlcv()
    # ... test logic ...
```

### 8.5 Code Style

- Python 3.10+ type hints encouraged
- Docstrings on all public functions (Google style preferred)
- Maximum line length: 120 characters
- Use `numpy` vectorized operations for performance-critical code
- Avoid global state; pass configuration explicitly

### 8.6 Running the Test Suite

```bash
# All tests
python -m pytest tests/ -q

# Specific test file
python -m pytest tests/test_kalman_strategy.py -v

# With coverage
python -m pytest tests/ --cov=spot_bot --cov-report=term-missing

# Skip slow tests
python -m pytest tests/ -q -m "not slow"
```

### 8.7 Commit Messages

Use conventional commits format:

```
feat: add dual-stream quantile signal mode
fix: correct feature index alignment after valid_mask filter
docs: add walk-forward validation tutorial
test: add regression test for limit-maker fill logic
refactor: extract cost model into standalone module
```

### 8.8 Pull Request Checklist

- [ ] All existing tests pass (`python -m pytest tests/ -q`)
- [ ] New code has docstrings
- [ ] Any new public API is documented in this file or README.md
- [ ] No hardcoded file paths (use `tmp_path` fixture for tests)
- [ ] No `freq='H'` / `freq='T'` (use lowercase `'h'` / `'min'`)
- [ ] CI workflow (`eval-metrics.yml`) passes

---

*For questions, please open a GitHub issue or reach out to the maintainers.*
