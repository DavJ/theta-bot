# Theta Bot - Complex-Time Theta Transform Trading Engine

A predictive trading engine grounded in Complex Consciousness Theory (CCT) and Unified Biquaternion Theory (UBT). It uses Jacobi theta functions for market prediction and applies complex-time (C) and log-time phase (psi_logtime) representations to detect regimes and manage risk. The canonical production path follows a Spot Bot 2.0 (long/flat) architecture with pluggable alpha strategies: mean reversion first, then Kalman mean/trend.

## Status

**Offline execution and evaluation implemented; a profitable production strategy is not established.**

**Binding constraint from 2026-10-07: owned spot funds only.** Leverage, borrowing,
margin, shorts, derivatives, leveraged tokens and martingale loss-doubling are
excluded. The planner rejects its legacy short switch; recorded fills cannot
borrow cash or sell unowned inventory. Live orders require verified spot markets
and sufficient free spot funds, including fees. Unknown information rejects the
order. See [the spot-only policy](docs/evaluation/SPOT_ONLY_POLICY.md).

The 2026-10-09 fixed momentum-refinement round tests six changes to the existing
daily spot momentum control. Capped inverse-volatility redistribution improves
the descriptive full-history net return from +103.91% to **+110.26%**, with the
same 50%/25% target caps and a conservative drawdown bound of -18.62% (-19.32%
under doubled fees/slippage). Fresh-account returns are +21.04% in 2025 and
+12.02% in January–September 2026; the continuous later return is +35.45%.
It executes 448 fills over 57 months, versus 432 for control. **This is an
exploratory comparison, not a passed development-frozen improvement screen:**
development selects the BTC regime filter, which subsequently trails control
and returns -0.04% in the 2026 fresh account. All 71 paths reconcile each 4h
close; none doubles in a rolling 7/30/90-day window. Live/default settings stay
unchanged. See [all refinements, costs and rejected variants](docs/evaluation/THETABOT_MOMENTUM_REFINEMENT_2026-10-09.md)
and [the fixed protocol](docs/evaluation/MOMENTUM_REFINEMENT_RESEARCH_PLAN.md).

The fixed 4h spot-flow comparison uses 171 checksum-verified BTC/ETH/BNB monthly
archives and 31,212 bars, with prior-bar signals, shared cash and ordinary/doubled
costs. **All six new momentum/breakout/reversion/flow-blend variants lose after
costs; none improves the existing spot candidate.** The development-frozen
control still returns +19.59% in 2025, +10.18% in January–September 2026 and
+103.91% on one continuous 2022–September 2026 account, with a −18.54%
conservative drawdown bound. No variant doubles its continuous account in a
rolling 7/30/90-day window. All 67 account paths reconcile every 4h close; no
live/default strategy is enabled. See [all flow results, costs and doubling counts](docs/evaluation/THETABOT_SPOT_FLOW_2026-10-07.md)
and [the fixed spot-flow protocol](docs/evaluation/SPOT_FLOW_RESEARCH_PLAN.md).

The fixed external-signal experiment adds five-day Ridge/boosting forecasts,
forex/equities/oil/rates, Alternative.me Fear & Greed and settled funding/taker
flow. Each available external feature has past-only 0–2 /3–9 /10–20 calendar-day
windows to represent a distributed market response. Refit every 20 days with
a 126-day training-weight half-life; publication latency is handled separately.
No extra Kalman/EKF filters these lag weights. All nine candidates were replayed
with ordinary and doubled costs, with development selection frozen first.
**The new models did not improve the existing momentum candidate.** Macro Ridge
returned +0.99% in 2025 and +6.86% in January–September 2026, versus momentum's
+19.59% /+10.18%. None of the five source additions improved forecast MSE under
the prescribed uncertainty test. Every new model exceeded the 20% conservative
drawdown bound on the continuous account. Macro/sentiment snapshots lack
first-release vintages, and Fear & Greed is a composite index rather than news
NLP. See [all source, algorithm and lag results](docs/evaluation/THETABOT_EXTERNAL_SIGNALS_2026-10-06.md)
and [the fixed experiment](docs/evaluation/EXTERNAL_SIGNAL_RESEARCH_PLAN.md).
This does not reject a mining-energy mechanism: WTI in a joint macro group is
not an isolated test of miners' electricity costs. Regional electricity and
gas data need their own publication-aware, BTC-focused experiment.

The user-defined research drawdown budget is now **20%**. A fixed seven-variant
comparison adds trailing covariance volatility sizing and a peak-preserving
cushion budget to the existing BTC/ETH/BNB momentum allocation. The development
selection is the 50% aggregate target cap: **+19.59% net in 2025**, **+10.18% in
January–September 2026**, and +103.91% over one continuous 2022–September 2026
account. Its close-only drawdown is −16.98%; the conservative daily high/low
portfolio drawdown bound is **−18.54%**, or −19.24% with doubled fees/slippage.
All periods were previously known, so passing this historical screen does not
establish a new confirmed alpha. A fixed artificial 40% simultaneous open gap
raises its drawdown bound to −27.30%; the 20% budget is not a guaranteed loss cap.
No risk settings or live/default sizing are enabled by this experiment. See the
[complete risk-budget results](docs/evaluation/THETABOT_RISK_BUDGET_2026-10-06.md)
and [the fixed 20% protocol](docs/evaluation/RISK_BUDGET_RESEARCH_PLAN.md).

The earlier fixed experiment adds a shared-cash spot portfolio on BTC, ETH and BNB,
weekly rebalancing, daily risk exits, actual market fills and a 30% aggregate target cap.
The 30-day momentum/inverse-volatility candidate returns **+11.45% in 2025** and
**+6.16% in January–September 2026**, after fees, slippage and spread. It is an exploratory
candidate identified after comparison: the development-frozen four-horizon allocation
returns only **+1.11% and −0.88%**, and fails the economic screen. No live orders were placed.
See [the complete portfolio and execution evidence](docs/evaluation/THETABOT_EXECUTION_PORTFOLIO_2026-10-06.md)
and the [fixed protocol](docs/evaluation/EXECUTION_PORTFOLIO_RESEARCH_PLAN.md).

An additional position-size sensitivity holds that exploratory model fixed and
replays 30/50/75/100% aggregate caps, using spot cash only. At the 100% target cap,
the historical net returns are +41.64% in 2025 and +19.07% in January–September 2026;
the continuous 2022–September 2026 account has a −31.34% maximum drawdown.
Higher exposure increases both profit and loss; it does not establish a new alpha
advantage or change the live/default configuration. See the
[complete sizing and adverse-period evidence](docs/evaluation/THETABOT_EXPOSURE_2026-10-06.md).

This round also corrects round-trip slippage, cost-per-turnover units, actual hysteresis
bounds and a one-candle delay in completed-day signals. Use `--execution-policy market`
in paper/replay/backtest to compare execution at the known decision price. Live exchange
execution continues to use `--order-type`. Earlier reports below preserve the results
of their earlier implementation; their execution metrics are superseded by the latest replay.

The 2026-10-06 audit fixes full-dataset percentile leakage, same-bar sizing/volatility leakage,
retroactive timeout fills, duplicated slippage charges, duplicated regime budgets, lost cost basis,
and Sharpe annualization. The regression suite and chronological screening run in CI.

On checksum-verified BTCUSDT 1h archives for April–September 2026, the candidate selected on
development data (`kalman_mr_dual`) returns **+0.82% net on the July–September holdout**.
Development return is **−0.74%**, one of three holdout subperiods is negative, and a 30% initial
allocation buy-and-hold returns **+8.88%** on the same holdout. The current paper screen fails.
These results use 1000 USDT, a 30% maximum target exposure, 0.1% fees per fill, 5 bps slippage,
and 2 bps spread. Open positions are marked to market, not forcibly liquidated.

See [the reproducible evaluation](docs/evaluation/THETABOT_2026-10-06.md).
The synthetic research results below are not evidence of production profitability.

The expanded comparison adds momentum, EMA trend, close-channel breakout, range reversion,
two exposure ensembles, calibrated forecast averaging and correlation-aware Kalman fusion.
On 2022–2024 development data the frozen winner is Kalman fusion, but it loses **0.87% in
2025** and **2.86% in January–September 2026** after costs. Breakout is the strongest observed
2025 candidate (**+2.31%**, then **+0.94%** in 2026 diagnostics); this exploratory comparison
does not establish a fresh confirmed trading edge. New strategies can exit losing inventory
and enforce the exposure cap at decision prices, overriding turnover hysteresis.
See [all 11 approaches, costs, forecast errors and reproducible results](docs/evaluation/THETABOT_MULTI_2026-10-06.md).

The strategies are available through `spot_bot.run_live --strategy`, including
`ensemble_equal`, `ensemble_regime`, `forecast_mean`, and `kalman_fusion`. Daily forecast
calibration requires at least 126 completed training outcomes and earlier indicator warmup;
use enough history (for example 10,000 hourly candles). Shorter fusion histories stay in cash.
EKF is not needed for the implemented linear measurement model. No live mode is enabled.

```bash
# Offline example with enough history; no exchange orders
python -m spot_bot.run_live --mode backtest --strategy kalman_fusion \
  --csv-in data/raw/BTCUSDT_1h_2024_2025.csv --limit-total 20000 --timeframe 1h \
  --fee-rate 0.001 --slippage-bps 5 --spread-bps 2 --max-exposure 0.3 \
  --out-summary bench_out/kalman_fusion_summary.json
```

For portfolio research, download the three symbols' daily archives with
`scripts.download_binance_archive --timeframe 1d` (at most 24 months per call), then run
`python -m scripts.research_execution_portfolio --help`. The allocation and replay APIs
are `spot_bot.portfolio.trend.TrendPortfolio` and
`spot_bot.backtest.portfolio.run_portfolio_backtest`. They use completed-day targets,
shared cash, core fee accounting and sell-before-buy execution; they do not connect
to an exchange. Optional `risk=PortfolioRisk(...)` supplies completed-return
covariance and/or an account cushion controller through `spot_bot.portfolio.risk`.
Risk reductions run at known daily opens; account peaks are never reset after
losses. Run `python -m scripts.research_risk_budget --help` to reproduce the fixed
20% experiment. The replay reports close-only drawdown, sampled open/close
drawdown and a conservative intraday bound separately; daily highs/lows only
affect the next day's sizing.

## Reproduce the current evaluation

```bash
pip install -r requirements.txt
python -m pytest -q tests

# Existing real dataset; runs fully offline
python -m spot_bot.evaluate --out bench_out/evaluation

# Complete public archive months; SHA-256 checked, no credentials
python -m scripts.download_binance_archive --start-month 2026-04 --end-month 2026-09 \
  --out data/raw/BTCUSDT_1h_2026_04_09.csv
python -m spot_bot.evaluate --csv data/raw/BTCUSDT_1h_2026_04_09.csv \
  --data-source binance_archive --out bench_out/evaluation_recent
```

The evaluator selects one existing strategy on the first 60% of observations, freezes that
choice, and tests it on the remaining 40% with historical feature warmup and fresh cash.
It writes `summary.json`, `report.md`, and holdout equity/trades CSVs. Add
`--require-paper-pass` to return a failing exit status when any screening check fails.
Passing the historical screen does not enable live execution.

## Quick Start (Spot Bot 2.0)

```bash
pip install -r requirements.txt

# Dryrun live (features + intent, no orders)
python -m spot_bot.run_live --mode dryrun --symbol BTC/USDT --timeframe 1h --limit-total 1000 --csv-out bench_out/live_features.csv --csv-out-mode features --strategy meanrev

# Backtest single CSV
python -m spot_bot.run_backtest --csv path/to/ohlcv.csv --strategy kalman --kalman-mode meanrev --slippage-bps 1

# Benchmark across pairs/psi modes
    python -m bench.benchmark_strategies --limit-total 8000 --timeframe 1h --out bench_out/strategies.csv --plots-dir bench_out/plots
```

## Fast backtest from CSV

Run the live entrypoint in backtest mode to process a local OHLCV CSV without downloading from Binance:

```bash
python -m spot_bot.run_live --mode backtest --csv-in data/btc_1m.csv --timeframe 1m --strategy kalman_mr_dual \
  --psi-mode scale_phase --psi-window 512 --rv-window 120 --fee-rate 0.001 --slippage-bps 5 --max-exposure 0.30 \
  --out-equity bench_out/btc_equity.csv --out-trades bench_out/btc_trades.csv --out-summary bench_out/btc_summary.csv
```

## Spot Bot 2.0 runners (long/flat)

- Regime smoke-test (logs optional):  
  `python -m spot_bot.run_regime --csv path/to/BTCUSDT_1h.csv --db bot.db`
- Backtest with simple fees/slippage and plots:  
  `python -m spot_bot.run_backtest --csv path/to/BTCUSDT_1h.csv --slippage-bps 1 --save-plots plots/`
- Live loop dryrun (no trades, logs optional):  
  `python -m spot_bot.run_live --mode dryrun --symbol BTC/USDT --timeframe 1h --limit-total 2000 --db bot.db --cache data/latest.csv`
- Paper trading (simulated fills at close, restartable from DB):  
  `python -m spot_bot.run_live --mode paper --db bot.db --initial-usdt 1000 --fee-rate 0.001 --max-exposure 0.3`
- Live execution (market orders) requires explicit opt-in flag:  
  `python -m spot_bot.run_live --mode live --i-understand-live-risk --db bot.db --symbol BTC/USDT --timeframe 1h`
- Live execution with limit maker orders (post-only):  
  `python -m spot_bot.run_live --mode live --i-understand-live-risk --db bot.db --symbol BTC/USDT --timeframe 1h --order-type limit_maker --maker-offset-bps 1 --max-spread-bps 15 --order-validity-seconds 120 --maker-fee-rate 0.0001 --taker-fee-rate 0.001`

Note: If running as a script instead of module mode, use `PYTHONPATH=.` to resolve imports.

### Limit Maker Orders

Limit maker orders are post-only orders that provide liquidity rather than taking it. They offer several advantages:

- **Lower fees**: Maker orders typically have lower fees (or even rebates) compared to taker orders
- **Better execution price**: Orders are placed at a favorable price (slightly better than current best bid/ask)
- **No immediate impact**: Orders don't execute immediately, avoiding market impact

**Key parameters:**
- `--order-type limit_maker`: Enable limit maker execution (default: `market`)
- `--maker-offset-bps 1.0`: Place orders 1 basis point better than current best bid/ask (default: 1.0)
- `--order-validity-seconds 60`: Cancel orders older than 60 seconds (default: 60)
- `--max-spread-bps 20.0`: Reject orders if spread exceeds 20 basis points (default: 20.0)
- `--maker-fee-rate 0.0001`: Fee rate for maker orders (default: same as --fee-rate)
- `--taker-fee-rate 0.001`: Fee rate for taker/market orders (default: same as --fee-rate)

**Important notes:**
- Limit maker orders return status `OPEN` and don't update portfolio balances until filled
- Orders are automatically cancelled if they remain unfilled beyond `order-validity-seconds`
- Wide spreads (beyond `max-spread-bps`) will reject order placement to avoid poor execution
- Paper and dryrun modes still use simulated execution (not limit orders)

### Manual Testing Steps

1. **Download real market data:**
   ```bash
   python download_market_data.py --symbol BTCUSDT --interval 1h --limit 2000
   ```

2. **Validate data quality:**
   ```bash
   python validate_real_data.py --csv real_data/BTCUSDT_1h.csv
   ```

3. **Run comprehensive tests on multiple pairs (including PLN):**
   ```bash
   # Full test with all pairs (USDT and PLN)
   python test_biquat_binance_real.py
   
   # Quick test mode (fewer pairs and horizons)
   python test_biquat_binance_real.py --quick
   
   # Skip download and use existing data
   python test_biquat_binance_real.py --skip-download
   ```
   
   This generates comprehensive HTML and Markdown reports in `test_output/`:
   - `comprehensive_report.html` - Interactive HTML report
   - `comprehensive_report.md` - Markdown summary
   
   **Features:**
   - Tests multiple trading pairs: BTC, ETH, BNB, SOL, ADA (with USDT and PLN)
   - Multiple test horizons (1h, 4h, 8h, 24h)
   - Strict walk-forward validation (NO data leaks)
   - Comprehensive performance metrics and visualizations
   - Data leak verification checks

4. **Run predictions:**
   ```bash
   python theta_predictor.py --csv real_data/BTCUSDT_1h.csv --window 512
   ```

5. **Run control tests:**
   ```bash
   python theta_horizon_scan_updated.py --csv real_data/BTCUSDT_1h.csv --test-controls
   ```

6. **Optimize hyperparameters:**
   ```bash
   python optimize_hyperparameters.py --csv real_data/BTCUSDT_1h.csv
   ```

## Performance on Synthetic Data

| Metric | h=1 | h=4 | h=8 |
|--------|-----|-----|-----|
| Correlation | 0.492 | 0.193 | -0.053 |
| Hit Rate | 65.6% | 56.4% | 46.1% |
| Sharpe Ratio | 14.24 | 6.70 | -0.82 |
| p-value | <10⁻⁸⁹ | 0.042 | n.s. |

**Note:** Real market data will likely show lower performance. Correlation r > 0.05 and hit rate > 52% is meaningful for real markets.

## Core Components

### Trading System

- **theta_basis_4d.py** - 4D orthonormalized Jacobi theta basis generation
- **theta_transform.py** - Forward and inverse theta transforms
- **theta_predictor.py** - Walk-forward prediction with no lookahead bias (v9 with biquaternion drift model)
- **theta_horizon_scan_updated.py** - Resonance scanning and control tests
- **generate_test_data.py** - Synthetic data generation for testing

### Theta Predictor v9 – Biquaternion Drift Model

The latest version of the theta predictor introduces advanced features while maintaining strict walk-forward causality:

**Key Features:**
- **Biquaternionic Time Base** (τ = t + jψ): Represents theta coefficients as Θ_k = a_k + i*b_k + j*c_k + k*d_k, where j is an imaginary quaternionic unit
- **Fokker-Planck Drift Term**: Captures macro bias with A_t = β₀ + β₁ * tanh(EMA₁₆(r_t))
- **PCA Regime Detection**: Adaptive regime detection via PCA clustering for trend/mean-reverting markets

**Usage:**
```bash
# Basic v9 prediction with all features
python theta_predictor.py --csv data.csv \
    --enable-biquaternion \
    --enable-drift \
    --enable-pca-regimes \
    --window 512 --horizons 1 4 8

# Standard prediction (backward compatible)
python theta_predictor.py --csv data.csv --window 512
```

**Expected Performance:**
- Correlation: 0.10–0.20 on real market data
- Hit rate: 52–56%
- Sharpe improvement: > +0.3 vs v8
- Maintains full UBT theoretical consistency

The v9 model restores the weak but real predictive edge observed in earlier biquaternionic implementations while maintaining strict walk-forward causality and no data leakage.

**Evaluation Status:**  
✅ Initial predictivity evaluation completed - see **[V9_EVALUATION_SUMMARY.md](V9_EVALUATION_SUMMARY.md)** for detailed findings.
- Evaluation tools: `evaluate_v9_predictivity_simple.py` and `evaluate_v9_predictivity.py`
- Results show mixed performance on mock data
- Further testing on real market data recommended

## Production Preparation Tools

- **download_market_data.py** - Download real market data from Binance
- **validate_real_data.py** - Comprehensive data validation and quality checks
- **optimize_hyperparameters.py** - Grid search for optimal parameters
- **production_readiness_check.py** - Automated end-to-end validation
- **quick_start.py** - One-command testing pipeline
- **test_biquat_corrected.py** - Test corrected biquaternion implementation
- **test_biquat_binance_real.py** - Comprehensive test on real Binance data with multiple pairs
- **tools/eval_metrics.py** - Evaluation script for computing performance metrics (earnings, correlation, hit rate)
- **evaluate_v9_predictivity_simple.py** - V9 vs V8 predictivity comparison (recommended)
- **evaluate_v9_predictivity.py** - Comprehensive v9 feature analysis

## Documentation

- **[PRODUCTION_PREPARATION.md](PRODUCTION_PREPARATION.md)** - Complete guide for preparing bot for production (START HERE!)
- **[V9_EVALUATION_SUMMARY.md](V9_EVALUATION_SUMMARY.md)** - Evaluation of v9 algorithm predictivity
- **[IMPLEMENTATION_SUMMARY.txt](IMPLEMENTATION_SUMMARY.txt)** - Implementation details and validation results
- **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)** - Detailed synthetic data test results
- **[CTT_README.md](CTT_README.md)** - Technical documentation and theory
- **[TEST_BINANCE_METRICS.md](TEST_BINANCE_METRICS.md)** - Evaluation metrics documentation (earnings, correlation, hit rate)

## Requirements

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
requests (optional, for downloading data)
```

## Architecture

The system implements complex-time dynamics: τ = t + iψ

Where:
- t = chronological time
- ψ = hidden phase component (psychological time)

Using Jacobi theta functions:
```
Θ(q, τ, φ) = Σ e^{iπn²τ} e^{2πinqφ}
```

This creates a modularly invariant, quasi-periodic structure for capturing temporal market patterns.

### Dual-Stream Theta + Mellin Model

A new advanced model architecture that combines two complementary representations:

**Architecture:**
- **Theta Stream**: Rolling window theta basis projections processed via GRU for temporal dynamics
- **Mellin Stream**: Mellin transform features capturing scale-invariant frequency characteristics
- **Gating Fusion**: Learned gating mechanism to adaptively combine both streams
- **PyTorch Backend**: Optional PyTorch implementation with automatic fallback to sklearn baseline

**Usage with Walk-forward Validation:**
```bash
# Run with dual-stream model (requires walk-forward pipeline)
cd theta_bot_averaging
python -m theta_bot_averaging.validation.walkforward configs/dual_stream_example.yaml
```

**Configuration Parameters:**
- `model_type: "dual_stream"` - Enable dual-stream architecture
- `signal_mode: "threshold"` - Signal generation mode (default: "threshold")
  - `"threshold"`: Fixed bps thresholds for long/short signals (default)
  - `"quantile"`: Quantile-based signals (95th percentile long, 5th percentile short per fold)
- `theta_window: 48` - Rolling window for theta basis (48 hours recommended)
- `theta_q: 0.9` - Theta basis decay parameter
- `theta_terms: 8` - Number of theta coefficients
- `mellin_k: 16` - Mellin frequency samples
- `torch_epochs: 50` - Training epochs (if PyTorch available)

**Signal Generation Modes:**

The system supports two signal generation modes for evaluation:

1. **Threshold Mode (Default)**: Uses fixed bps thresholds
   - Long signal: `predicted_return > threshold_bps / 10000`
   - Short signal: `predicted_return < -threshold_bps / 10000`
   - Neutral: otherwise

2. **Quantile Mode (Evaluation Only)**: Ranks predictions per fold
   - Long signal: `predicted_return > 95th percentile`
   - Short signal: `predicted_return < 5th percentile`
   - Neutral: otherwise
   - Purpose: Tests whether model can rank opportunities independent of absolute magnitudes

To use quantile mode, set `signal_mode: "quantile"` in your config YAML or use `--signal-mode quantile` with evaluation scripts.

**Example:**
```bash
# Evaluate with quantile-based signals
python evaluate_dual_stream_predictivity.py --signal-mode quantile

# Or use a config file
python scripts/run_walkforward.py --config configs/dual_stream_quantile.yaml
```

**Key Features:**
- **No Lookahead**: Strict causal feature extraction (validated by tests)
- **Graceful Fallback**: Uses BaselineModel when PyTorch unavailable
- **Tested**: Comprehensive test suite for shapes, causality, and end-to-end integration

See `configs/dual_stream_example.yaml` for a complete configuration example.

## Validation Status

✅ Mathematical Properties:
- Orthonormality: error < 10⁻¹⁵ (machine precision)
- Hermitian symmetry: error < 10⁻¹⁸
- Energy conservation: validated
- Eigenvalue spectrum: properly normalized

✅ Code Quality:
- Code review: Passed
- CodeQL security scan: 0 vulnerabilities
- Error handling: Robust
- Documentation: Comprehensive

⏳ Production Testing:
- Real market data testing: **IN PROGRESS**
- Control tests: **PENDING**
- Hyperparameter optimization: **PENDING**
- Paper trading: **NOT STARTED**

## Next Steps

1. ✅ Complete implementation
2. ✅ Validate on synthetic data
3. 🔄 **Test on real market data** (current phase)
4. ⏳ Run control tests (permutation, noise)
5. ⏳ Optimize hyperparameters for real data
6. ⏳ Paper trading validation
7. ⏳ Live deployment with risk management

## Warning

⚠️ **This is research software.** Always:
- Test thoroughly with paper trading first
- Use proper risk management
- Monitor performance continuously
- Be prepared to disable if performance degrades

Past performance (especially on synthetic data) does not guarantee future results.

## License

See repository license file.

## References

- Complex Consciousness Theory (CCT)
- Unified Biquaternion Theory (UBT)
- Jacobi Theta Functions
- Modular Forms and Market Dynamics

### Viewing the Theta Function Evaluation (Atlas Evaluation)  
  
A new evaluation document has been added in `theta_bot_averaging/paper/atlas_evaluation.tex`. This LaTeX paper demonstrates the correctness and physical consistency of the implemented Jacobi theta functions (\theta_1–\theta_4) and the choice of nome \(q\) and imaginary time.  
  
To view the evaluation, compile the LaTeX file with a TeX engine such as `pdflatex`:  
  
```bash  
cd theta_bot_averaging/paper  
pdflatex atlas_evaluation.tex  
```  
  
After compilation, open the resulting `atlas_evaluation.pdf` in your preferred PDF viewer to read the detailed analysis.
