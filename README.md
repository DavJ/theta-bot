# Theta Bot — spot trading research

Python research software for cryptocurrency spot trading. The project contains
Jacobi theta and Mellin features, Kalman and forecast-fusion models, and daily
momentum portfolio experiments with shared-cash execution accounting.

**A profitable production strategy is not established.** Offline replay,
chronological evaluation and execution guards are implemented. The positive
historical benchmark uses 30-day momentum and inverse 60-day volatility;
theta-derived forecasts and additional signal sources remain research modules.

## Current results

Latest spot experiments: **2026-10-09**; optional long-option model: **2026-10-10**. These are historical simulations,
with previously studied periods, rather than newly unseen confirmation data.

Each account starts with 1,000 USDT. Full history is one continuous account from
January 2022 through September 2026; the later comparison is a separate continuous
account from January 2025 through September 2026. Net results include modeled
0.1% fees per fill, 5bps slippage and half a 2bps spread. Cost stress doubles fees
and slippage; timing stress adds one 4h signal delay. Terminal inventory is marked
to market. CAGR is the annualized compounded return over full history.

| Research policy | Full net | Full CAGR | Later net | Full DD: nominal | Full DD: doubled costs | Full DD: +4h delay |
|---|---:|---:|---:|---:|---:|---:|
| Original BTC/ETH/BNB momentum (`spot_control`) | +103.91% | +16.19% | +31.75% | -18.54% | -19.24% | -20.20% |
| Capped allocation (`capped_control`) | +110.26% | +16.95% | +35.45% | -18.62% | -19.32% | -20.37% |
| Ranked 30-day momentum (`ranked_m30`) | +115.49% | +17.55% | +40.85% | -20.81% | -21.40% | -20.15% |
| 14/30/60-day consensus (`consensus`) | +85.03% | +13.84% | +27.96% | -19.32% | -19.96% | -17.91% |
| Latest development-frozen broad policy (`broad_top5_vol20`) | +54.54% | +9.60% | -0.92% | -22.01% | -23.13% | -22.99% |

DD is the conservative 4h high/low portfolio drawdown bound, including continuous
account peaks. It assumes unfavorable intrabar/cross-asset ordering and is not
an exact tick-level measurement. The historical selection budget is **20%**;
it is not a guaranteed future maximum loss.

Capped allocation has higher nominal profit than the original benchmark, but its
delayed drawdown exceeds the budget. Ranked momentum exceeds the nominal budget.
Consensus stays within the budget in these three historical scenarios but earns
less than capped allocation and does not pass the fixed improvement screen.
None of these comparisons establishes a production-ready strategy.

The latest experiment expands to **15 assets from a dated December 2021 cohort**,
with historical liquidity selection, weekly ranks, old/new Terra separation and
documented MATIC/POL continuity. Development freezes `broad_top5_vol20` before
later economic replay. All six broader policies trail both controls on full
nominal net profit and exceed 20% full drawdown. The frozen policy also fails
cost/time stress; a material improvement is not supported. Its descriptive paired
block interval is not confirmatory statistical significance on reused data.

The multiscale and broad-universe studies each reconcile **104 account paths** at
every 4h close. Both previous controls are reproduced. No policy in either study
doubles its account in a rolling 7/30/90-day window. These overlapping historical
windows do not estimate independent future doubling probabilities.

- [Broad-universe results, source audits and all rejected variants](docs/evaluation/THETABOT_BROAD_UNIVERSE_2026-10-09.md) · [JSON evidence](docs/evaluation/THETABOT_BROAD_UNIVERSE_2026-10-09.json) · [Fixed protocol](docs/evaluation/BROAD_UNIVERSE_RESEARCH_PLAN.md)
- [Multiscale, cost and timing results](docs/evaluation/THETABOT_MULTISCALE_SPOT_2026-10-09.md) · [Fixed protocol](docs/evaluation/MULTISCALE_SPOT_RESEARCH_PLAN.md)

The optional paid-option model initially reserves 10% of the original account,
with monthly purchases capped at 20% of that sleeve's current cash, including
fees. It never replenishes option losses from spot. Four fixed signals on inverse
and USDC contracts are compared with full-capital spot and idle-cash controls.
For 1,000 quote units, options do not improve full-period profit; inverse minimum
lots prevent every buy. For 10,000 units, the development-frozen inverse directional
5%-OTM policy models **+314.31%** full-account net versus **+108.82%** spot, and
**+53.44%** versus **+35.31%** on the separately funded 2025–2026 account.

This model uses historical trades with premium markups, not executable asks;
held-option marks and drawdowns are synthetic. Full profit falls to **+132.15%**
with a 4h delay. Cost stress gives **+236.23%**, but modeled DD reaches **20.14%**
and fails the 20% budget. Three options contribute 80.04% of net option P&L.
Retained wins grow the sleeve to 54.62% of the final account, so its initial 10%
funding is not a permanent exposure cap. This is an exploratory result with no
live eligibility; it does not establish executable option profitability.

- [All option models, stress results and limitations](docs/evaluation/THETABOT_LONG_OPTIONS_2026-10-10.md) · [JSON evidence](docs/evaluation/THETABOT_LONG_OPTIONS_2026-10-10.json) · [Closed-option audit](docs/evaluation/THETABOT_LONG_OPTIONS_2026-10-10_CLOSED_OPTIONS.csv) · [Fixed protocol](docs/evaluation/LONG_OPTIONS_RESEARCH_PLAN.md)

## Binding trading constraints

Only owned spot cash and inventory may be traded. Leverage, loans, margin,
financial shorts, futures/options, leveraged tokens and martingale loss-doubling
are excluded from live execution. A 2026-10-10 user-authorized exception permits
offline modeling of fully paid long calls/puts in an isolated cash sleeve; the
research switch defaults off and has no order adapter. Long options still contain
economic leverage. See the [fixed option protocol](docs/evaluation/LONG_OPTIONS_RESEARCH_PLAN.md).
Cash and inventory must remain nonnegative, including fees. The
planner rejects its legacy short switch; impossible fills are rejected. Exchange
execution requires verified spot-market information and sufficient free funds.

Latest portfolio studies use 50% aggregate and 25% per-asset target caps, Monday
ordinary rebalances, daily inactivity/cap reductions, a 1pt band and a 10-USDT
minimum fill. Positions can drift between decisions. An absent quote cannot
execute a trade. Source/indicator prices stay absent; explicitly flagged replay
sentinels value held unquoted inventory at zero conservatively, and any such path
fails the candidate's quality gate.

Offline research and passing tests do not authorize deployment, orders, changing
live defaults or merging a PR. The research portfolio policies above are
available through `scripts.research_*`; they are not the runner's live defaults.
See [the binding spot-only policy](docs/evaluation/SPOT_ONLY_POLICY.md) and
[repository constraints](AGENTS.md).

## Setup and offline evaluation

Run commands from the repository root. Python 3.12 is used in CI. Install the
project's dependencies from [requirements.txt](requirements.txt); PyTorch is
optional for the dual-stream research model.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
python test_eval_metrics.py

# Bundled dataset; fully offline, no exchange orders
python -m spot_bot.evaluate --out bench_out/evaluation
```

`spot_bot.evaluate` selects among its existing single-asset strategies on the
first 60% of observations, freezes the choice and tests the remaining 40% with
historical feature warmup and fresh cash. It writes `summary.json`, `report.md`
and holdout equity/trade CSVs. The bundled dataset has been used before.
`--require-paper-pass` returns a failing status when screening fails. A passing
report does not enable live execution.

For checksum-verified recent public spot data, without account credentials:

```bash
python -m scripts.download_binance_archive --start-month 2026-04 --end-month 2026-09 \
  --out data/raw/BTCUSDT_1h_2026_04_09.csv
python -m spot_bot.evaluate --csv data/raw/BTCUSDT_1h_2026_04_09.csv \
  --data-source binance_archive --out bench_out/evaluation_recent
```

This single-asset evaluation is separate from the multi-asset research accounts
in the results table. It does not reproduce their portfolio returns.

## Reproduce the portfolio experiments

The downloaders use official public spot archives and verify published SHA-256
checksums. Source manifests retain URLs, hashes, identity rules and absence audits.
The broad downloader verifies 853 monthly archives, 15 pre-period proofs and one
corroborating daily artifact. Conflicting or unverified source rows stop the study.

```bash
# BTC/ETH/BNB controls, refinements and multiscale comparison
python -m scripts.download_spot_flow
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_momentum_refinement
python -m scripts.write_momentum_refinement_report
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_multiscale_spot
python -m scripts.write_multiscale_spot_report

# Fixed historical 15-asset cohort; reuses the three-asset cache when available
python -m scripts.download_spot_universe
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_broad_universe
python -m scripts.write_broad_universe_report

# Optional paid-option MODEL: historical trade proxies, synthetic held marks
python -m scripts.download_long_options
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_long_options --options-enabled
python -m scripts.write_long_options_report
```

Each fixed study persists its development selection before later economic replay.
Completed-day targets execute at the next open, with sells before buys and shared
cash. Raw CSV/cache/manifests live under `data/raw/`; replay selections, targets,
quote flags and equity/fill CSVs live under `artifacts/` (Git ignored). Durable
report/JSON evidence is under `docs/evaluation/`. Reports link their exact protocols.

For an individual existing strategy on a local CSV:

```bash
python -m spot_bot.run_live --mode backtest --strategy kalman_fusion \
  --csv-in data/raw/spot_flow/BTCUSDT_4h.csv --timeframe 4h --limit-total 20000 \
  --execution-policy market --fee-rate 0.001 --slippage-bps 5 --spread-bps 2 \
  --max-exposure 0.3 --out-summary bench_out/kalman_fusion_summary.json
```

`dryrun` emits intent without orders; `paper` uses simulated fills and requires a
SQLite `--db`. Exchange execution code also exists and has an explicit live opt-in.
Its presence does not establish a profitable bot. Limit-maker simulation lacks
order-book queue and partial-fill evidence. Use `--help` for runner parameters.

## Architecture and research components

| Area | Role |
|---|---|
| [spot_bot/core/](spot_bot/core/) | Owned-funds planning, simulated execution, fee accounting and portfolio state |
| [spot_bot/backtest/](spot_bot/backtest/) | Single/shared-cash replay, quote flags, drawdown and independent ledger reconciliation |
| [spot_bot/portfolio/](spot_bot/portfolio/) | Completed-day momentum, allocation, covariance budgets, liquidity and weekly ranks |
| [spot_bot/strategies/](spot_bot/strategies/) | Mean reversion, Kalman, trend/breakout and forecast-fusion strategies |
| [spot_bot/features/](spot_bot/features/) | Causal features and external-source availability handling |
| [spot_bot/execution/](spot_bot/execution/) | Exchange execution adapters and spot-account checks |
| [theta_bot_averaging/](theta_bot_averaging/) | Theta/Mellin models and walk-forward predictor experiments |
| [scripts/](scripts/) | Checked data downloaders, fixed economic studies and report writers |

The theta research includes orthonormal Jacobi bases, complex-time/phase
representations, biquaternion drift and PCA regimes. The dual-stream model
combines theta and Mellin features with optional PyTorch and a sklearn fallback.
These implementations and their mathematical tests do not establish market profit.

The historical [V9 predictivity evaluation](V9_EVALUATION_SUMMARY.md) reports mixed,
weak results on mock data. It does not support the former README's expected
real-market correlation/hit-rate ranges or a validated theta trading edge.
Synthetic diagnostics remain in [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md).
Signed forecasts are research outputs; executable spot targets are nonnegative.
Legacy long/short P&L experiments are outside the current trading policy.

Further technical material: [theta/CTT documentation](CTT_README.md),
[dual-stream configuration](configs/dual_stream_example.yaml),
[implementation notes](IMPLEMENTATION_SUMMARY.txt) and
[Atlas theta-function paper](theta_bot_averaging/paper/atlas_evaluation.tex).
These are research references, not production-profit certificates.

## Earlier experiments and evidence

| Study | Evidence |
|---|---|
| Momentum allocation refinements | [Results](docs/evaluation/THETABOT_MOMENTUM_REFINEMENT_2026-10-09.md) |
| Executed spot-flow signals | [Results](docs/evaluation/THETABOT_SPOT_FLOW_2026-10-07.md) |
| Forex/equities/oil/rates, sentiment and variable-lag forecasts | [Results](docs/evaluation/THETABOT_EXTERNAL_SIGNALS_2026-10-06.md) |
| 20% historical risk-budget comparison | [Results](docs/evaluation/THETABOT_RISK_BUDGET_2026-10-06.md) |
| Portfolio execution and sizing sensitivity | [Execution](docs/evaluation/THETABOT_EXECUTION_PORTFOLIO_2026-10-06.md) · [Sizing](docs/evaluation/THETABOT_EXPOSURE_2026-10-06.md) |
| Multi-model forecasts and correlation-aware Kalman fusion | [Results](docs/evaluation/THETABOT_MULTI_2026-10-06.md) |
| Single-asset chronological audit | [Results](docs/evaluation/THETABOT_2026-10-06.md) |

Higher historical exposure and legacy execution metrics in earlier reports are
archived comparisons. The latest corrected replay accounts for fees, spread,
slippage, continuous peaks and delayed completed-day signals. Repeatedly studied
periods must not be relabeled as fresh holdouts. The external-source report also
discloses unavailable first-release vintages and the composite nature of Fear &
Greed; its joint oil/macro test does not isolate miners' electricity costs.

## Verification and remaining work

The 2026-10-09 research code passed **590 regression tests and 8 legacy metrics
tests**. Its [CI run](https://github.com/DavJ/theta-bot/actions/runs/37995546773)
verified pandas 3.0.6 / NumPy 2.5.3, entry-point syntax, chronological evaluation
and evidence upload. Tests cover future-prefix invariance, token identity,
publication clocks, sparse quotes, owned balances and independent accounting.
CI success proves these checks passed; it does not prove future profit or replace
a security review.

Further work requires a fixed protocol, source/publication audits and a candidate
that survives costs, latency and the 20% historical risk budget. Confirmation
then needs genuinely unseen or forward paper evidence. Rejected variants and all
cost/risk failures remain documented. No production rollout follows automatically.

Additional data guidance: [real-data runbook](docs/data/REAL_DATA_RUNBOOK.md) and
[data protocol](docs/data/DATA_PROTOCOL.md). Older
[production-preparation notes](PRODUCTION_PREPARATION.md) are historical tooling
references and do not supersede the current status or binding trading constraints.
