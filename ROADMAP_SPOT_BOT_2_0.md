# Spot Bot 2.0 Roadmap (Long/Flat)
_C denotes complex-time carrier features that blend price/volume with an imaginary log-time axis to capture market tempo; psi_logtime denotes log-time phase features used to detect regimes for risk gating._

## Phase A — Infrastructure MVP (no trading)
- **Goal:** Stand up reproducible scaffolding that ingests data, computes C + psi_logtime features, emits regime decisions, and captures logs/telemetry without sending orders.
- **Deliverables (files/modules):** Data loaders in `spot_bot/` for CSV/exchange inputs, feature computation modules for C and psi_logtime, a regime decision engine that outputs long/flat intents, dry-run runner/backtest harness, baseline configs under `configs/`, and structured logging/metrics sinks.
- **Exit criteria (measurable):**
  - Deterministic dry-run producing regime decision files and logs on reference pairs (e.g., BTC/USDT, ETH/USDT) with identical SHA256 output checksums and config hashes across two consecutive runs, saved as CSV/Parquet under `reports/regimes/`.
  - CI/pytest suite passes with the pipeline wired.
  - Ops docs exist for running the pipeline end-to-end without trading.
- **What NOT to do:** Do not place live or paper trades, do not introduce sizing/execution layers, do not add new alpha models, and do not run unfenced experiments beyond feature/regime validation.

## Phase B — Combination #1 (primary): Mean Reversion + Risk Gating (C + psi_logtime)
- **Goal:** Deploy the primary long/flat mean reversion alpha gated by non-directional risk detection using C + psi_logtime regimes.
- **Deliverables (files/modules):** Mean reversion strategy module producing intents, gating integration with the regime engine, configs/backtests demonstrating gated vs. ungated performance, monitoring/reporting templates in `reports/`, and switchable flags for enabling the strategy.
- **Exit criteria (measurable):**
  - Controlled backtests on reference pairs show gated Sharpe/Calmar improvements over ungated MR (absolute Sharpe uplift ≥ +0.2, e.g., 1.0 → 1.2), matching current production guardrails.
  - Max drawdown ≤ 15%.
  - Backtests span at least one year of hourly data (≥ 10k bars).
  - Shadow-mode runs produce stable gated intents for a week without missing data or crashes.
- **What NOT to do:** Do not add Kalman/trend logic, do not enable leverage/shorting, do not bypass gating, and do not onboard new data domains (sentiment, options) yet.

## Phase C — Combination #2 (later): Kalman mean/trend + Risk Sizing (C + psi_logtime)
- **Goal:** Introduce Kalman-based mean/trend estimation with risk sizing layered on top of C + psi_logtime gating while staying long/flat.
- **Deliverables (files/modules):** Kalman filter signal module, risk sizing component (position scaling/hold duration), configs to toggle between MR and Kalman strategies, comparative backtests, and shadow-mode logging for sizing decisions.
- **Exit criteria (measurable):**
  - Kalman + sizing backtests outperform MR baseline on stability metrics.
  - Max drawdown ≤ 12% (tighter than Phase B to reflect sizing discipline).
  - Realized volatility no higher than MR baseline.
  - Exposure caps defined in risk configs (e.g., max 1.0x notional, position steps ≤ target limits) documented under `configs/risk_limits.yaml` are met.
  - Shadow-mode sizing logs are complete and reproducible across runs.
- **What NOT to do:** Do not activate sentiment/LSTM inputs, do not change execution venue or order types, do not relax gating/risk limits, and do not promote to live trading until sizing controls are signed off.

## Phase D — Optional wave: Sentiment / LSTM (postponed)
- **Goal:** Capture the future optional path for sentiment/LSTM augmentation while keeping it explicitly out of the production path until Phases B/C are stable.
- **Deliverables (files/modules):** A design/backlog note outlining data sources, feature hooks, and evaluation plan; placeholder interfaces behind feature flags (no active wiring).
- **Exit criteria (measurable):** Documented go/no-go checklist and required metrics for enabling sentiment/LSTM; feature flags remain off and no sentiment data pulled in production runs.
- **What NOT to do:** Do not integrate external sentiment feeds, do not train/deploy LSTM models, and do not adjust production configs to depend on this wave until a formal green light.
