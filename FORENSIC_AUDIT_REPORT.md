# FORENSIC AUDIT REPORT

**Repository:** DavJ/theta-bot  
**Branch audited:** `copilot/forensic-audit-historical-experiments`  
**Audit date:** 2026-04-03  
**Methodology:** Code-trace based. All claims verified against actual scripts, not README summaries.  
**Guiding principle:** A result is not "working" unless the evaluation path is trusted. A result is not "tradable" unless costs are accounted for.

---

## EXECUTIVE SUMMARY

The theta-bot repository contains two main systems:

1. **Spot Bot 2.0** — A long/flat spot trading engine with four pluggable strategy variants, a unified realistic cost model, and a production-capable live runner. **No preserved backtest results exist** for any strategy variant.

2. **Theta Bot Averaging** — A research prediction engine based on Jacobi theta functions, biquaternion representation, and Mellin transforms. Its results cluster into two distinct categories:
   - **Synthetic data:** Strong signals (corr 0.4–0.8). These are artifacts of testing against perfectly periodic sine waves.
   - **Real data:** Near-random performance (corr ~0.02–0.03, hit rate ~50%), negative PnL after fees.

**The single most critical finding:** The only real-data evaluation that was reported (the `eval_metrics.py` run producing −$39 to −$478 losses) did **not** test the biquaternion model. It used a naive momentum fallback by mistake. This means we have **no trusted real-data evaluation of the core prediction model** anywhere in the repository.

**The second most critical finding:** Multiple experiments labelled as "real Binance data" tests actually ran on mock/simulated data because the Binance API was unavailable in the CI environment. All performance claims from those runs should be treated as synthetic results.

---

## SECTION 1: EVALUATOR TRACE AUDIT

### 1.1 eval_metrics.py — INVALID EVALUATOR

**Location:** `eval_metrics.py` (~line 387)

**What it claims to test:** Biquaternion model predictivity on real Binance data.

**What it actually tests:** Naive momentum (predicted return = previous period's return, `pct_change().shift(1)`). This is a silent fallback that was inserted instead of the actual model prediction call.

**Evidence:** `ANALYSIS_POOR_RESULTS.md` documents this explicitly:

> *"The eval_metrics.py script is NOT using the biquaternion model predictions. Instead, it's using a naive momentum strategy as a fallback."*

**Impact:** The reported results in `test_output/eval_summary.md` (BTCUSDT −$39 without fees, −$400 with fees; corr 0.025; hit rate 52.7%) are **not evidence** about the biquaternion model. They are evidence that naive momentum loses money on 1H BTC — which is expected and uninteresting.

**Classification:** INVALID_EVAL. Discard entirely.

### 1.2 test_biquat_binance_real.py — CLAIMED REAL, ACTUALLY MOCK

**What it claims:** "Test of the corrected biquaternion implementation on real Binance data."

**What it actually tested:** Mock/simulated data. `ZAVERECNA_ZPRAVA_CZ.md` confirms:

> *"Reálná data z Binance nebyla nahrána — síťové připojení k api.binance.com je blokováno."*  
> ("Real Binance data was not loaded — network connection to api.binance.com is blocked.")

The script generates a warning but continues with mock data. Any claim that "the bot was tested on real data" from this script is false.

**Classification:** SYNTHETIC_ONLY (despite the label).

### 1.3 theta_eval_biquat_corrected.py — VALID EVALUATOR (SYNTHETIC DATA)

**What it tests:** Biquaternion complex-pair ridge regression on synthetically generated data.

**Code path:** Strictly causal. Walk-forward split: model fitted on `[t-window, t)`, predicts at `t`. Standardization computed only on fit window. No future data visible. Code-trace confirms this.

**What it cannot tell us:** Whether there is any edge in real markets. The synthetic data consists of perfectly periodic sine waves at known frequencies — the Jacobi theta basis is mathematically designed to capture these exactly. High correlation on this synthetic data is **circular**: the model was built to recover the exact signal type used in the test.

**Classification:** VALID for what it tests. Results meaningless for real-market inference.

### 1.4 robustness_suite_v3_oos.py → theta_eval_hbatch_biquat_max.py — VALID FRAMEWORK

**What it tests:** Multi-symbol OOS correlation and hit rate; causal chronological 70/30 split; lag-h leakage detection.

**Code path:** Causal. No lookahead detected in ridge regression logic. Leakage flag computed (compares correlation at shifted predictions vs ground truth autocorrelation).

**Limitation:** No preserved output in the repository. The script expects data from `prices/` directory (via `make_prices_csv.py`) which is not present. Results have never been captured.

**Classification:** VALID framework, UNCLEAR_NEEDS_RERUN for results.

### 1.5 evaluate_dual_stream_real.py — VALID FRAMEWORK (UNRUN)

**What it tests:** Baseline logistic regression vs dual-stream model on real `data/BTCUSDT_1H_real.csv.gz`.

**Code path:** Walk-forward split; uses real compressed data file that **does exist** in the repository. This is the **only** evaluation that would use actual real market data and a credible model.

**Limitation:** No preserved output found in the repository. Has not been run with results captured.

**Classification:** VALID framework, UNCLEAR_NEEDS_RERUN.

### 1.6 evaluate_v9_predictivity.py — VALID FRAMEWORK, WEAK RESULTS

**What it tests:** V9 algorithm (biquaternion + Fokker-Planck drift + PCA regime) vs V8 baseline.

**Data:** Mock market data (labelled as "realistic simulations"). Not real exchange data.

**Results (from `V9_EVALUATION_SUMMARY.md`):**

| Horizon | V8 Correlation | V9 Correlation | Δ |
|---|---|---|---|
| 1h | 0.0189 | 0.0176 | −0.001 |
| 4h | 0.0125 | 0.0153 | +0.003 |

Both models are effectively random on the test data. V9 is 3× slower.

**Classification:** VALID framework, REAL_DATA_WEAK (using mock data).

---

## SECTION 2: VALIDITY CLASSIFICATION SUMMARY

| Experiment | Evaluator | Status | Justification |
|---|---|---|---|
| eval_metrics.py results | INVALID | INVALID_EVAL | Tested naive momentum, not the model |
| Biquaternion on synthetic | theta_eval_biquat_corrected.py | SYNTHETIC_ONLY | Causal; but synthetic periodic data only |
| Biquaternion on "Binance" | test_biquat_binance_real.py | SYNTHETIC_ONLY | Actually ran on mock data |
| V9 algorithm | evaluate_v9_predictivity.py | REAL_DATA_WEAK | Weak results on mock data |
| Dual-stream (synthetic) | evaluate_dual_stream_predictivity.py | SYNTHETIC_ONLY | Optimized config shows edge; standard config does not |
| Dual-stream (real) | evaluate_dual_stream_real.py | UNCLEAR_NEEDS_RERUN | Script valid; no output captured |
| OOS robustness suite | robustness_suite_v3_oos.py | UNCLEAR_NEEDS_RERUN | Framework valid; no output captured |
| Spot Bot strategies (all) | fast_backtest.py | UNCLEAR_NEEDS_RERUN | Code sound; no results preserved |
| Theta 4D basis | theta_basis_4d.py | SYNTHETIC_ONLY | Mathematical, not financial, validation |
| Chronofactor/UBT theory | None | DEAD_END | No financial evaluation exists |

---

## SECTION 3: REAL VS SYNTHETIC SEPARATION

### Confirmed synthetic / mock data (results are NOT real-market evidence):

- `FINAL_REPORT.md` results (18.1% corr improvement): synthetic periodic sines
- `DUAL_STREAM_EVALUATION_REPORT.md` results (+41% Sharpe): synthetic data
- `V9_EVALUATION_SUMMARY.md` results: mock realistic data
- `test_output/eval_summary.md` results: mock data + wrong model (naive momentum)
- `BINANCE_DATA_TEST_REPORT.md` results: mock data (API blocked)
- All `ZAVERECNA_ZPRAVA_CZ.md` results: mock data

### Confirmed real-data assets in repository:

- `data/BTCUSDT_1H_real.csv.gz` — real Binance hourly data, source confirmed
- Scripts designed to use it: `evaluate_dual_stream_real.py`, `eval_biquat_binance.py`

### Real-data evaluations that have been run:

- `REAL_DATA_ANALYSIS.md` documents real data results (from an earlier run with real Binance data):
  - BTC/USDT: hit rate 0.4814, corr −0.005 → below random
  - ETH/USDT: hit rate 0.4951, corr 0.056 → near random
  - SOL/USDT: hit rate 0.5083, corr 0.042 → marginal
  - Average: hit rate ~0.49, corr ~0.01 → essentially random

This is the most honest and credible real-data result in the repository. It shows that the biquaternion basis model (Jacobi theta functions) fails to produce any meaningful edge on liquid crypto hourly data.

---

## SECTION 4: GROSS VS NET EDGE RECONSTRUCTION

### 4.1 Biquaternion / Theta family (research models)

**Gross edge (synthetic):** Strong — corr 0.4–0.8 at horizons 1–8. But this is circular with the synthetic data generator.

**Gross edge (real data):** Near zero — corr −0.01 to 0.06, averaging ~0.01. No meaningful directional signal.

**Net edge (real data):** Negative. Even the weak gross signal is wiped out by taker fees (0.1% per side) at typical hourly trading turnover.

**Verdict:** No gross edge on real data → no net edge. The path from biquaternion math to financial profitability is not established.

### 4.2 Spot Bot strategies (Mean Reversion, Kalman, Dual Kalman, LSTM+Kalman)

**Gross edge:** Unknown — no preserved backtest output exists for any strategy.

**Net edge:** Unknown — however, the cost model is realistic (fees + slippage + spread + hysteresis). The ROADMAP_SPOT_BOT_2_0.md targets Sharpe ≥ 1.2 with max drawdown ≤ 15% on gated mean reversion. This is a credible if optimistic target.

**Hysteresis impact:** Documented to reduce turnover by 40–60%, which significantly reduces the fee drag on the mean-reversion strategy.

**Verdict:** Cannot assess without running the backtests. The infrastructure is the most credible in the repo for producing valid results.

### 4.3 Dual-Stream model

**Gross edge (synthetic, optimized):** Corr +41%, Sharpe +40% over logistic baseline. On optimized hyperparameters with 800+ samples.

**Gross edge (synthetic, standard):** Baseline (logistic regression) outperforms dual-stream on all metrics (corr, Sharpe, return). Dual-stream is worse with standard config.

**Gross edge (real data):** Not assessed — no preserved output from `evaluate_dual_stream_real.py`.

**Net edge:** Unknown.

**Verdict:** High sensitivity to hyperparameters. Wins only in the best-case synthetic config. Inconclusive until real-data test is run.

---

## SECTION 5: WINNING SIGNAL HYPOTHESIS EXTRACTION

Based on the audited experiments, here is where the strongest evidence exists (or doesn't):

### 5.1 Complex time / biquaternion / Jacobi theta basis
**Evidence for edge:** None on real data. Strong on synthetic, but this is explained by the match between synthetic data generation (periodic sines) and model design (theta functions). This is a circular test.

**Verdict:** The deep model hypothesis (complex time gives market edge) is **unsubstantiated by real evidence**. The mathematical elegance is real; the market relevance is not demonstrated.

### 5.2 Phase features (log-phase, scale-phase, concentration C)
**Evidence:** These are used as regime features in the spot bot. Their financial value is embedded in the spot bot backtest (EXP-008 to EXP-011), which has not been run to produce preserved output.

**Verdict:** Plausible regime detection utility. Cannot assess without running spot bot backtests.

### 5.3 Execution improvements (hysteresis, limit maker)
**Evidence:** Hysteresis is documented to reduce turnover by 40–60%. Limit maker (post-only orders) reduces fees significantly (from ~0.1% taker to ~0% maker). These are real improvements that apply regardless of the underlying signal.

**Verdict:** The strongest demonstrated improvements in the repo are **execution-side**, not prediction-side. Hysteresis and limit maker orders are legitimate cost-reduction mechanisms.

### 5.4 Regime / risk gating (S score, risk state)
**Evidence:** The spot bot regime engine (S score, RV-based risk states) gates position sizing. Theoretically sound — avoiding trading in high-volatility regimes reduces drawdown and turnover.

**Verdict:** Plausible. Needs backtest results to confirm.

### 5.5 Mean reversion (EMA z-score)
**Evidence:** Classical, well-established in crypto mean reversion literature. The spot bot implementation is clean and cost-aware.

**Verdict:** Most likely to show gross edge on real data. The signal is simple and well-understood, unlike the biquaternion family.

---

## SECTION 6: RERUN RECOMMENDATIONS

In priority order, from highest to lowest expected information value:

### RERUN-1 (Critical): Spot Bot Backtest — All Four Strategies
**Why:** These are the only experiments with realistic cost models and a production-grade execution engine. Any of the four strategies (Mean Reversion, Kalman, Dual Kalman, LSTM+Kalman) could show net edge.

**Minimal rerun:**
```bash
# Use the backtest runner with 2+ years of hourly data
python -m spot_bot.run_backtest \
  --strategy mean_reversion \
  --symbol BTCUSDT \
  --timeframe 1h \
  --fee-rate 0.001 \
  --slippage-bps 5 \
  --spread-bps 2
```

Run all four strategies. Compare Sharpe, max drawdown, win rate, and net PnL. Preserve output.

### RERUN-2 (High): Dual-Stream on Real Data
**Why:** `evaluate_dual_stream_real.py` is the most credible research-track evaluator, uses real data from the repo, and has never produced captured output.

**Minimal rerun:**
```bash
python theta_bot_averaging/eval/evaluate_dual_stream_real.py \
  --output-dir results/dual_stream_real_$(date +%Y%m%d)
```

Compare baseline vs dual-stream. Apply full cost model.

### RERUN-3 (Medium): OOS Robustness Suite on Real Multi-Symbol Data
**Why:** `robustness_suite_v3_oos.py` is the most rigorous evaluation framework in the repo (multi-symbol, causal OOS split, leakage flag). Never run to completion with results captured.

**Minimal rerun:** Requires downloading 6-symbol data first. Then run `robustness_suite_v3_oos.py` with the default parameters from the script's README.

### RERUN-4 (Low): Biquaternion Model on Real Data (Fix eval_metrics.py)
**Why:** Fix the evaluator bug, then run with the actual biquaternion model on `data/BTCUSDT_1H_real.csv.gz`. This will confirm or deny the weak real-data results from `REAL_DATA_ANALYSIS.md`.

**Expected outcome:** Likely confirms weak/no edge, consistent with prior real-data results. Worth confirming before abandoning.

---

## SECTION 7: WHAT THE README AND DOCS OVERCLAIM

The following documentation claims are not supported by the code-trace evidence:

1. **"18.1% improvement in prediction correlation"** (`FINAL_REPORT.md`) — This is on synthetic periodic data. It is not a real-market result.

2. **"41% correlation improvement"** (`DUAL_STREAM_EVALUATION_REPORT.md`) — On synthetic optimized config only. Standard config shows the opposite.

3. **"Bot tested on real Binance data"** (multiple reports) — All Binance tests ran on mock data due to API unavailability.

4. **"Production-ready"** (`PRODUCTION_SUMMARY.md`) — The research models (biquaternion, dual-stream) have no real-data backtest results. The spot bot strategies have a production-capable engine but also no preserved backtest output.

5. **V9 evaluation claims** (`V9_EVALUATION_SUMMARY.md`) — V9 performed no better than V8 and was 3× slower. The summary does not clearly state that both models are essentially random.

---

## FINAL EXECUTIVE ANSWERS

**1. What actually worked best?**  
Nothing has been demonstrated to work on real market data. The execution improvements (hysteresis, limit maker) are the only validated improvements — they reduce friction regardless of signal quality. The spot bot mean reversion strategy is the most credible untested candidate.

**2. What only looked good because of bad evaluation?**  
The biquaternion model looked good only on synthetic data (circular: designed to capture exactly the signal type in the test). The eval_metrics.py report looked bad because of wrong code path (tested naive momentum, not the model). Both are evaluation failures in opposite directions.

**3. What was profitable before fees but not after fees?**  
Cannot be determined — the only gross-edge results exist on synthetic data. No real gross edge has been established to then test against fees.

**4. What is the single best candidate to revive first?**  
**Spot Bot Mean Reversion with Dual Kalman gating (EXP-010)**, run as a realistic backtest on 2+ years of hourly data with the existing unified cost model. This candidate has: a realistic cost model, a production-capable engine, regime gating, hysteresis for turnover reduction, and a well-understood signal type. It is the most likely to produce meaningful results quickly.

**5. What should we stop spending time on?**  
- Biquaternion basis / Jacobi theta functions as primary prediction signals — no real-data edge after extensive development effort.
- V9 algorithm (biquaternion + drift + PCA) — slower than V8, no better results.
- Chronofactor/UBT derivations — pure theory with no evaluation path.
- 4D theta basis mathematical validation — already done; does not constitute financial evidence.
- Any further synthetic data benchmarks — the synthetic data type is too favorable to these models to be informative.
