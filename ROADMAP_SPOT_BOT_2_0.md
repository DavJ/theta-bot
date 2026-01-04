# Spot Bot 2.0 Roadmap (Long/Flat)

## Phase A — Infrastructure MVP (no trading)
- Stand up modular scaffolding for data providers, feature pipeline, regime engine, strategies, sizing, execution, and runners.
- Validate wiring with dry-run backtests and live pipelines without placing trades.
- Establish logging/telemetry hooks and configuration surfaces for downstream phases.

## Phase B — Mean Reversion + Risk Gating (C + psi_logtime)
- Activate a mean-reversion strategy that produces long/flat intents gated by regime classification.
- Incorporate psi_logtime features and gating logic into the feature pipeline and regime engine.
- Run controlled backtests to confirm signal stability and gating effectiveness before live shadow mode.

## Phase C — Kalman + Risk Sizing (C + psi_logtime)
- Replace or complement the strategy layer with a Kalman filter-driven signal that remains long/flat.
- Introduce risk sizing on top of psi_logtime features to modulate exposure while maintaining gating discipline.
- Graduate from shadow to production once execution, sizing, and monitoring meet reliability thresholds.
