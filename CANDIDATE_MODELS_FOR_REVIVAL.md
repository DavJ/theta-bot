# CANDIDATE MODELS FOR REVIVAL

**Repository:** DavJ/theta-bot  
**Audit date:** 2026-04-03  
**Based on:** FORENSIC_AUDIT_REPORT.md, EXPERIMENT_REGISTRY.md, FEE_IMPACT_ANALYSIS.md

---

## Guiding Criteria

A candidate is shortlisted only if it meets all of the following:

1. The evaluator path is trusted (or can be trusted with minimal fixes)
2. There is a plausible mechanism for gross edge beyond synthetic artifacts
3. The cost model is realistic or can be made realistic without architectural changes
4. Reviving it does not require inventing new models — only running existing code correctly

Candidates are ranked by the quality of evidence for real (non-synthetic) edge.

---

## CANDIDATE 1: Spot Bot Mean Reversion with Dual Kalman Gating

**Experiment reference:** EXP-010  
**Code path:** `spot_bot/strategies/meanrev_dual_kalman.py` + `spot_bot/backtest/fast_backtest.py`  
**Evaluation script:** `spot_bot/run_backtest.py` or `spot_bot/backtest_runner.py`

### Why It Is Promising

- **Signal:** EMA z-score mean reversion is a well-established, widely documented edge in liquid crypto markets at hourly timeframes. It is not an exotic claim.
- **Gating:** Dual Kalman filter provides regime detection (trend vs mean-reverting regime) via SNR-based confidence. This is a principled way to avoid trading against strong trends.
- **Cost model:** The existing unified cost model (`spot_bot/core/cost_model.py`) is the most realistic in the repository (fees + slippage + spread + hysteresis + min notional). The backtest uses the same code path as the live runner, reducing implementation divergence.
- **Execution:** Limit maker support exists (`LIMIT_MAKER_IMPLEMENTATION.md`). Switching to maker orders reduces fee drag by 8–20 bps per trade.
- **Turnover control:** Hysteresis layer documented to reduce turnover by 40–60% while maintaining Sharpe.

### Whether Edge Was Gross-Only or Net-Positive

**Unknown.** No backtest results have been preserved. However, the infrastructure is the right one to answer this question.

### What Likely Killed It If It Failed

There is no evidence it failed — it simply was never run to a captured result. The ROADMAP targets Sharpe ≥ 1.2, max drawdown ≤ 15%, which is ambitious but plausible for a gated mean reversion strategy.

If results are negative when run, the most likely cause would be:
- **High turnover:** Mean reversion can generate many small trades. Fee drag at taker rates (20+ bps round-trip) would destroy it without hysteresis.
- **Regime mismatch:** Strong trend periods (2021 BTC bull run, 2022 bear) would generate losses unless regime gating excludes them.

### Minimal Rerun to Validate

```bash
# Download or use existing data
# Run backtest with conservative defaults
python -m spot_bot.run_backtest \
  --strategy meanrev_dual_kalman \
  --symbol BTCUSDT \
  --timeframe 1h \
  --fee-rate 0.001 \
  --slippage-bps 5 \
  --spread-bps 2 \
  --limit-total 2000

# Repeat for ETHUSDT cross-validation
# Compare vs mean_reversion (simple) and kalman strategies
```

Capture equity curve, trade list, Sharpe, max drawdown, win rate, turnover.

### Revival Category

**Production signal** — if net Sharpe > 1.0 on real data with realistic costs  
**Research alpha** — if gross edge is present but net Sharpe < 1.0 (use to guide further development)  
**Discard** — if no gross edge on real data

### Confidence in Revival

**Medium.** The strategy and infrastructure are sound. The signal type (mean reversion) has empirical support in the literature. The cost model is realistic. The main risk is that crypto markets have become more efficient and shorter mean reversion cycles may be arbitraged away.

---

## CANDIDATE 2: Spot Bot Mean Reversion (Simple — Baseline Comparison)

**Experiment reference:** EXP-008  
**Code path:** `spot_bot/strategies/mean_reversion.py` + `spot_bot/backtest/fast_backtest.py`  
**Evaluation script:** `spot_bot/run_backtest.py`

### Why It Is Promising

- Simpler signal than Candidate 1 — useful as a baseline to determine how much the Dual Kalman adds
- Same realistic cost model
- EMA z-score mean reversion is well-understood; backtest results would be interpretable

### Minimal Rerun to Validate

```bash
python -m spot_bot.run_backtest \
  --strategy mean_reversion \
  --symbol BTCUSDT \
  --timeframe 1h \
  --fee-rate 0.001 \
  --slippage-bps 5 \
  --spread-bps 2
```

### Revival Category

**Regime/risk overlay baseline** — primarily useful as a reference point for measuring the value added by Dual Kalman gating (Candidate 1) and other enhancements.

### Confidence in Revival

**Medium-low.** Simple mean reversion without regime gating may not survive strong trend periods. Value depends on whether gating (Candidate 1) improves it meaningfully.

---

## CANDIDATE 3: Dual-Stream Model on Real BTCUSDT Data

**Experiment reference:** EXP-006  
**Code path:** `theta_bot_averaging/eval/evaluate_dual_stream_real.py`  
**Data:** `data/BTCUSDT_1H_real.csv.gz`

### Why It Is Promising

- Uses real market data that is already in the repository
- Walk-forward evaluation with causal split
- Partial cost model in the backtest engine (4 bps maker fee, 1 bp slippage, 0.5 bp spread)
- The script explicitly compares baseline logistic regression vs dual-stream — a controlled comparison
- No API access required (data is local)

### Whether Edge Was Gross-Only or Net-Positive

Unknown — the script has never been run to a captured result. The DUAL_STREAM_EVALUATION_REPORT.md results (synthetic) suggest that:
- Standard hyperparameters: dual-stream is *worse* than logistic baseline
- Optimized hyperparameters (800 samples): dual-stream is better

For real data, the baseline comparison will reveal whether either model has meaningful gross edge.

### What Likely Killed It If It Failed

High sensitivity to hyperparameters. The dual-stream model requires 800+ samples and specific hyperparameter tuning to outperform the logistic baseline. On real market data with weaker periodic structure, the theta/Mellin features may not provide lift over a simple baseline.

### Minimal Rerun to Validate

```bash
python theta_bot_averaging/eval/evaluate_dual_stream_real.py \
  --output-dir results/dual_stream_real_$(date +%Y%m%d)
```

If the script requires hyperparameter arguments, use the optimized config from the report (theta_window=72, n_terms=12, mellin_k=20, epochs=50, lr=5e-4).

### Revival Category

**Research alpha** — if baseline outperforms or ties, there is no reason to use the more complex dual-stream model. If dual-stream shows real-data gross edge, it could become an input feature for the spot bot's regime engine.

### Confidence in Revival

**Low-medium.** The real-data results from REAL_DATA_ANALYSIS.md (prior biquaternion run) showed near-random performance. The dual-stream model may do somewhat better due to learned feature fusion, but the underlying market data (1H BTC) is highly efficient and unlikely to show the periodic structure the model needs.

---

## CANDIDATES EXPLICITLY NOT SHORTLISTED

### V9 Algorithm (EXP-004)

**Why excluded:** V9 showed no improvement over V8 on mock data. It is 3× slower. There is no plausible mechanism by which the additions (Fokker-Planck drift term, PCA regime) would recover edge that is absent in the simpler biquaternion baseline.

**Verdict:** Do not revive.

### Biquaternion Basis / Theta Functions as Primary Prediction Signal (EXP-002, EXP-003)

**Why excluded:** Real-data results consistently show near-random performance (corr −0.005 to 0.056, average ~0.01 across multiple assets). The synthetic-data success is circular — the model is designed to capture sine waves and the synthetic data is sine waves. There is no evidence of edge on real markets after multiple attempts.

**Verdict:** Stop investing development time on the biquaternion basis as a standalone prediction signal. If it is to be used at all, it should be only as a secondary feature input to a downstream model (e.g., as a regime feature in the spot bot) — where its value would be isolated by ablation.

### LSTM + Kalman Hybrid (EXP-011)

**Why excluded:** LSTM adds significant complexity and overfitting risk in limited-data regimes (hourly crypto, 2000 bars). No evidence of edge. Should only be revisited if Candidate 1 (Dual Kalman) shows clear edge and LSTM provides measurable improvement on top.

**Verdict:** Lower priority than Candidates 1 and 2. Revisit only if simpler strategies show meaningful edge.

### Chronofactor / UBT Derivations (EXP-014)

**Why excluded:** No evaluation code exists. Pure theory. Cannot be revived without a complete implementation effort, which falls outside the scope of this audit.

**Verdict:** Preserve as reference. Do not implement until a concrete, testable hypothesis is written.

---

## Revival Priority Summary

| Priority | Candidate | Type | Action |
|---|---|---|---|
| **1 (Critical)** | Spot Bot Dual Kalman Mean Reversion (EXP-010) | Production signal | Run backtest immediately; preserve output |
| **2 (High)** | Spot Bot Simple Mean Reversion (EXP-008) | Baseline comparison | Run concurrently with #1 |
| **3 (Medium)** | Dual-Stream on Real BTCUSDT (EXP-006) | Research alpha | Run with real data; compare vs logistic baseline |
| **—** | V9 Algorithm (EXP-004) | Do not revive | Near-random; 3× overhead |
| **—** | Biquaternion as primary signal | Do not revive | No real-data edge after extensive effort |
| **—** | LSTM+Kalman (EXP-011) | Deprioritize | Only if #1/#2 show clear edge |
| **—** | Chronofactor / UBT theory | Preserve only | No evaluation path |

---

## Minimal Validation Plan for Top Candidates

### Step 1: Run Spot Bot Backtests (1 day of effort)

1. Obtain 2 years of BTCUSDT 1H OHLCV data
2. Run all four spot bot strategies using `spot_bot/backtest_runner.py`
3. Capture equity curves, trade logs, and summary metrics
4. Compare: Sharpe, max drawdown, win rate, turnover, net PnL

**Decision gates:**
- If mean reversion shows Sharpe < 0.5 on net-of-fee basis → investigate signal
- If Dual Kalman gating adds Sharpe ≥ +0.2 over ungated MR → confirm it adds value
- If any strategy shows Sharpe > 1.0 net-of-fees → candidate for paper trading

### Step 2: Run Dual-Stream Real-Data Eval (0.5 day of effort)

1. Run `evaluate_dual_stream_real.py` using existing `data/BTCUSDT_1H_real.csv.gz`
2. Apply full cost model via `theta_bot_averaging/backtest/engine.py`
3. Compare baseline logistic regression vs dual-stream under same conditions
4. If dual-stream gross corr > 0.05 on real data → worthy of integration into spot bot feature pipeline

### Step 3: Decision

Based on Steps 1 and 2:
- If spot bot strategies show net edge → prepare for paper trading, then live
- If dual-stream shows gross edge → consider integrating as a regime feature
- If neither shows edge → the repo requires a fundamentally different signal source, not more tuning of existing approaches
