# FEE IMPACT ANALYSIS

**Repository:** DavJ/theta-bot  
**Audit date:** 2026-04-03  
**Purpose:** Document what cost assumptions were used in historical experiments, classify cost realism, and identify where fees destroyed or would destroy edge.

---

## 1. Cost Model Architecture

### 1.1 Unified Cost Model (Spot Bot 2.0)

**Location:** `spot_bot/core/cost_model.py`

```python
def compute_cost_per_turnover(fee_rate, slippage_bps, spread_bps):
    return fee_rate + 2.0 * (slippage_bps / 10_000) + (spread_bps / 10_000)
```

**Components:**
- `fee_rate`: Taker fee rate (both entry and exit); default CLI: `0.001` (10 bps per side = 20 bps round-trip)
- `slippage_bps`: Market impact per leg; default CLI: `5.0` (5 bps per side = 10 bps round-trip)
- `spread_bps`: Bid-ask spread; default CLI: `2.0` (2 bps average)

**Round-trip cost at defaults:** 10+10+10+2 = ~22 bps per full position change  
*(fee 2×10bps + slippage 2×5bps + spread 2bps)*

This is applied in `spot_bot/core/engine.py` for all execution modes (live, paper, backtest) and in the hysteresis computation.

### 1.2 Theta Bot Averaging Backtest Cost Model

**Location:** `theta_bot_averaging/backtest/engine.py`

```python
def _apply_transaction_costs(position, fee_rate, slippage_bps, spread_bps):
    trades = position.diff().abs()
    fee_cost = trades * fee_rate
    slippage_cost = trades * (slippage_bps / 10_000.0)
    spread_cost = trades * (spread_bps / 10_000.0)
    return fee_cost + slippage_cost + spread_cost
```

**Defaults:** fee_rate=0.0004 (4 bps), slippage_bps=1.0, spread_bps=0.5

**Note:** These defaults are lower than the spot_bot CLI defaults. 4 bps maker fee is realistic on Binance if using maker/limit orders. 1 bp slippage is optimistic for 1H OHLCV-based simulation.

---

## 2. Cost Realism by Experiment

### EXP-001: eval_metrics.py (Naive Momentum)

| Cost Component | Applied? | Value |
|---|---|---|
| Trading fees | YES | 0.1% taker per side (200 bps round-trip) |
| Slippage | NO | Not modelled |
| Spread | NO | Not modelled |
| Turnover impact | NO | Not modelled |
| Min notional / lot rounding | NO | Not modelled |

**Cost realism: PARTIAL**

**Impact:**
- Without fees: −$39 USDT (small loss, reflecting weak momentum edge)
- With 0.1% taker fees: −$400 USDT (catastrophic — ~10× worse)
- This demonstrates that high-frequency naive momentum is completely destroyed by taker fees

**Note:** The fee rate used (0.1% per side) is the standard taker fee. This is realistic for market orders. However, the naive momentum model is invalid (INVALID_EVAL) so the cost impact is not informative about the real model.

---

### EXP-002: Biquaternion on Synthetic Data (theta_eval_biquat_corrected.py)

| Cost Component | Applied? | Value |
|---|---|---|
| Trading fees | NO | Not modelled |
| Slippage | NO | Not modelled |
| Spread | NO | Not modelled |
| Turnover impact | NO | Not modelled |

**Cost realism: NONE**

**Gross-to-net gap:** Cannot be computed. Reported correlation (0.413 at h=1) is a gross metric only.

**Estimate:** If this signal existed on real data with the same correlation (~0.4) and the model traded every bar:
- Gross Sharpe would be high
- But: 1H trading at 20 bps round-trip = 48 × 20 bps/day = 960 bps/day in fees alone → obviously unviable
- Position holding for multiple hours before flipping would be required

**Verdict:** Cost analysis not applicable until real-data edge is established. On synthetic data, cost impact is irrelevant.

---

### EXP-004: V9 Algorithm (evaluate_v9_predictivity.py)

| Cost Component | Applied? | Value |
|---|---|---|
| Trading fees | NO | Not modelled |
| Slippage | NO | Not modelled |
| Spread | NO | Not modelled |

**Cost realism: NONE**

**Gross metrics:**
- Correlation: 0.0176 at h=1 (essentially random)
- Hit rate: 43.9% (below 50%)

**Fee impact:** Irrelevant — no gross edge to preserve.

---

### EXP-005: Dual-Stream on Synthetic (evaluate_dual_stream_predictivity.py)

| Cost Component | Applied? | Value |
|---|---|---|
| Trading fees | NO | Not modelled |
| Slippage | NO | Not modelled |
| Spread | NO | Not modelled |

**Cost realism: NONE**

**Gross metrics (optimized config):**
- Sharpe: 5.55 (synthetic)
- Cumulative return: 6.16× (synthetic, 800 samples)

**These are gross-only metrics on synthetic data.** The cumulative return of 6.16× is entirely from the model's predictive power, not after realistic execution.

**Estimate on real data with costs:** If we assume real gross corr of ~0.05 (optimistic for real data):
- At 20 bps round-trip and hourly trading, cost drag would far exceed any realistic gross signal
- Net edge would be negative

---

### EXP-006: Dual-Stream on Real BTCUSDT (evaluate_dual_stream_real.py)

| Cost Component | Applied? | Value |
|---|---|---|
| Trading fees | YES (partial) | fee_rate=0.0004 (4 bps) default |
| Slippage | YES (partial) | slippage_bps=1.0 (1 bp per leg) |
| Spread | YES (partial) | spread_bps=0.5 |

**Cost realism: PARTIAL**

The backtest engine in `theta_bot_averaging/backtest/engine.py` does apply all three cost components. The default fee rate (4 bps maker) is realistic if using post-only limit orders. Slippage of 1 bp per leg is optimistic.

**No output captured** — cannot assess actual net impact.

---

### EXP-007: OOS Robustness Suite (robustness_suite_v3_oos.py)

| Cost Component | Applied? | Value |
|---|---|---|
| Trading fees | NO | Not modelled — OOS correlation metrics only |
| Slippage | NO | Not modelled |
| Spread | NO | Not modelled |

**Cost realism: NONE**

This evaluator measures gross predictivity only (OOS correlation, hit rate). It explicitly does not simulate a PnL-generating strategy.

**Implication:** Even if OOS corr comes back at 0.05–0.10, we would still need a full backtest with cost model to assess net edge.

---

### EXP-008 to EXP-011: Spot Bot Strategies (Backtests)

| Cost Component | Applied? | Value |
|---|---|---|
| Trading fees | YES | fee_rate configurable; default CLI 0.001 (10 bps/side) |
| Slippage | YES | slippage_bps configurable; default CLI 5.0 (5 bps/leg) |
| Spread | YES | spread_bps configurable; default CLI 2.0 |
| Hysteresis (turnover control) | YES | Adaptive threshold — suppresses small trades |
| Min notional / rounding | YES | trade_planner.py enforces min_notional |

**Cost realism: REALISTIC**

This is the most realistic cost model in the repository. The default CLI parameters are conservative:
- 10 bps taker fee per side (Binance spot standard without VIP levels)
- 5 bps slippage per leg (reasonable for $5k–$50k orders on BTC hourly)
- 2 bps spread (reasonable for BTC/USDT)

**Round-trip cost at defaults:** ~22 bps  
**Break-even gross edge per trade:** ~22 bps (position change must generate at least 22 bps net gain)

**Hysteresis impact on fees:**  
The hysteresis formula `delta_e_min = max(hyst_floor, hyst_k * cost * (rv_ref / rv_current))` ensures that small position adjustments below the threshold are suppressed. Documentation states 40–60% reduction in turnover. This proportionally reduces fee drag.

**Fee impact at 50% turnover reduction:**  
- Without hysteresis: ~22 bps × T trades/year
- With hysteresis: ~22 bps × 0.5T trades/year  
- This is a substantial real improvement

**No backtest output captured** — cannot compute actual annual fee drag or net returns.

---

## 3. Limit Maker Execution

**Location:** `LIMIT_MAKER_IMPLEMENTATION.md`, `spot_bot/core/executor.py`

The spot bot supports post-only limit orders (`--order-type limit_maker`). On Binance:
- Taker fee: 0.10% per side (standard)
- Maker fee: 0.00%–0.05% per side (VIP level dependent; 0% for large-volume accounts)

**Impact of switching to limit maker:**
- Reduces fee from ~10 bps/side to ~0–2 bps/side
- Round-trip fee savings: 8–20 bps per trade
- On 100 trades/year: 800–2000 bps = 8–20% saved annually

This is the single largest lever available for improving net returns, independent of signal quality.

---

## 4. Summary: Cost Realism Classification by Experiment

| Experiment | Fee | Slippage | Spread | Hysteresis | Classification | Key Insight |
|---|---|---|---|---|---|---|
| EXP-001 (eval_metrics.py) | YES | NO | NO | NO | PARTIAL | 0.1% taker fee destroyed naive momentum completely (−$39 → −$400) |
| EXP-002 (BQ synthetic) | NO | NO | NO | NO | NONE | No cost model |
| EXP-003 (BQ "Binance") | UNKNOWN | UNKNOWN | UNKNOWN | NO | UNKNOWN | Mock data + blocked API |
| EXP-004 (V9 mock) | NO | NO | NO | NO | NONE | No cost model |
| EXP-005 (Dual-stream synthetic) | NO | NO | NO | NO | NONE | No cost model |
| EXP-006 (Dual-stream real) | YES (4bps) | YES (1bp) | YES (0.5bp) | NO | PARTIAL | Realistic but optimistic on slippage |
| EXP-007 (OOS robustness) | NO | NO | NO | NO | NONE | Gross metrics only |
| EXP-008 (MR backtest) | YES (10bps) | YES (5bps) | YES (2bps) | YES | REALISTIC | Most realistic; no output preserved |
| EXP-009 (Kalman backtest) | YES | YES | YES | YES | REALISTIC | Most realistic; no output preserved |
| EXP-010 (Dual Kalman backtest) | YES | YES | YES | YES | REALISTIC | Most realistic; no output preserved |
| EXP-011 (LSTM+Kalman backtest) | YES | YES | YES | YES | REALISTIC | Most realistic; no output preserved |

---

## 5. Gross-to-Net Edge Ranking

Based on available evidence, experiments ranked from most to least promising for surviving realistic costs:

| Rank | Experiment | Gross Edge | Cost Model | Net Edge Estimate | Rationale |
|---|---|---|---|---|---|
| 1 | EXP-010: Spot Bot Dual Kalman | Unknown | REALISTIC | Unknown | Most sophisticated signal + best cost model + hysteresis |
| 2 | EXP-008: Spot Bot Mean Reversion | Unknown | REALISTIC | Unknown | Simple classical signal + best cost model + hysteresis |
| 3 | EXP-009: Spot Bot Kalman | Unknown | REALISTIC | Unknown | Adaptive leverage + best cost model |
| 4 | EXP-006: Dual-stream real | Unknown | PARTIAL | Unknown | Real data + partial cost model |
| 5 | EXP-007: OOS robustness suite | Unknown | NONE | Unknown | Best gross-edge evaluator; no cost model |
| 6 | EXP-002: Biquaternion synthetic | Strong (synthetic only) | NONE | Negative (estimated) | Synthetic edge; real-data corr near zero |
| 7 | EXP-005: Dual-stream synthetic | Strong (synthetic, optimized) | NONE | Negative (estimated) | Config-sensitive; no real-data evidence |
| 8 | EXP-004: V9 algorithm | Near-zero (mock) | NONE | Negative | Near-random gross edge |
| 9 | EXP-001: eval_metrics.py | Weak negative | PARTIAL | Strongly negative | Wrong model; costs destroyed it further |

---

## 6. Key Conclusions

1. **The most critical missing piece is running the Spot Bot backtests (EXP-008 to EXP-011) with preserved output.** These are the only experiments with realistic cost models and a credible strategy signal.

2. **The research models (biquaternion, dual-stream) have no cost model attached.** Even if gross edge were real (which it isn't on real data), we would need to add a cost model before drawing any conclusions about net edge.

3. **The largest mechanical improvement available is switching to limit maker execution.** This saves 8–20 bps per trade regardless of the underlying signal, potentially the difference between break-even and profitable.

4. **Hysteresis is a legitimate and documented improvement.** A 40–60% reduction in turnover translates directly to proportionally lower fee drag.

5. **Fee drag at 1H frequency is severe.** At 22 bps round-trip, a strategy needs to hold positions for multiple hours on average to break even. Any signal with average holding time < 3–4 bars will be fee-destroyed at standard taker rates.
