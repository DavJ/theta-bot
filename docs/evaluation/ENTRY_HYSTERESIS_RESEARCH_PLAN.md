# Fixed momentum-entry hysteresis experiment — 2026-10-25

The user requests materially higher net profitability. The existing 10,000-USDT
nominal control ledger attributes 89.31 of 857.55 USDT modeled fees/impact to
risk reductions and 768.24 to other fills. Therefore test ordinary entry churn
with unchanged risk/execution rules. This diagnosis uses the already published
control ledger, not new variant payoffs.

## Fixed candidate and diagnostic variants

The candidate is `entry_hysteresis`, fixed before all new economics. A market
becomes eligible only when its completed 30-day simple momentum exceeds 0.0064.
Eligibility then persists until momentum is nonpositive. Warmup requires 60
completed daily returns for inverse-volatility sizing. Eligibility is a signal
state, not actual inventory; missed Monday entries do not become fake fills.
Daily signal history is computed once from 2022 and is also available before
the independently funded later account starts. Warmup is never shortened.

The fixed 0.64% threshold equals twice the approximate nominal round-trip
allowance: 2*(2*(0.1% fee + 5bps slippage + half of a 2bps spread)). It is a
trailing-strength heuristic, not an estimate of future net edge, an exact
break-even quote or a 30-day holding instruction. Do not tune it for stress.

Compare exactly five policies:

- `guarded_edge`: unchanged positive-30-day-momentum control.
- `entry_hysteresis`: enter above 0.64%, leave at nonpositive momentum.
- `symmetric_hurdle`: eligibility only while momentum is above 0.64%; no memory.
- `two_day_confirmation`: two consecutive warmup-ready positive-momentum days
  to become eligible, then leave at nonpositive momentum.
- `hysteresis_confirmed`: two consecutive warmup-ready days above 0.64% to
  become eligible, then leave at nonpositive momentum.

Only the fixed candidate may pass the improvement screen; diagnostics do not
replace it after results. A reconstruction of `guarded_edge` must match the
existing capped-allocation target before economic replay. No new asset,
exposure multiplier, forecast, parameter grid or exit delay is introduced.

## Execution, dates and gates

Use identical verified BTCUSDT/ETHUSDT/BNBUSDT 4h archives from January 2022 to
September 2026. All dates/assets were studied previously; this is not an unseen
holdout. Load complete aligned data, recheck all 171 official archive checksums
and match archive identities to the previous cost/risk report.

Keep Monday ordinary rebalancing, 1pp band-edge execution, 10-USDT minimum,
nonnegative owned cash/inventory, 50% total/25% asset target caps and every-4h
actual-holdings volatility reductions. The same 15% requested annual volatility,
60 completed daily returns, 25% covariance diagonal shrinkage and missing-history
cash rule apply to all five. Target publication is at completed-day 20h labels;
the engine lags target and covariance one bar, with an additional bar in timing
stress. Existing exits, mandatory cap reductions and risk sales remain active.

Independently fund 1,000 and 10,000 USDT accounts. Development: 2022–2024;
later continuous: 2025–September 2026; full continuous: 2022–September 2026.
Nominal fee/slippage/spread: 0.1%, 5bps, 2bps. Cost stress doubles fees and
slippage. Also run one additional 4h signal/covariance lag and combined stress.
This yields 120 paths. Save an exclusive fixed development selection before
later/full economics. Reproduce all 24 archived control summaries. Accounts
are continuous inside each period; do not splice or add separate-period returns.

Historical profit improvement requires all accounts to reconcile; candidate
drawdown bounds within 20% in every path; positive candidate later/full net
under every stress; and strictly higher candidate net than control on later/full
nominal and combined stress, at both capital sizes. This is a descriptive screen
on reused history, not statistical confirmation or a future loss guarantee.

Separately diagnose the turnover mechanism: candidate modeled fees plus impact
and fill counts must be lower than control for later/full nominal and combined
stress at both capitals. More profit with higher costs can pass the profit
screen but does not support the proposed cost-saving mechanism. Disclose actual
exposure, realized daily volatility and conservative continuous drawdown.

## Evidence and scope

Publish frozen plan, source fingerprints, settings, runtime and source manifest
before new payoffs. Preserve every result and rejection, all published targets,
and ten complete 10,000-USDT nominal/combined full equity and trade ledgers.
Independently audit CSV fills against source opens, fees/impact, nonnegative
balances, NAV, continuous drawdown, lagged covariance and reported final economics.
Use the existing independent auditor because risk mathematics are unchanged.
Export checksummed JSON and reconstructable archive parts with all evidence.

Keep implementation outside spot_bot and all earlier frozen source files. Both
the 2026-10-16 prospective model and 2026-10-23 public-depth protocols must retain
their source fingerprints. The recent two order-book captures do not justify
lowering historical cost assumptions. Model fills, terminal marks and USDT
research do not establish account/EEA execution, actual fees or queue/depth
availability. Operating costs, taxes, custody and quote FX remain outside P&L.
No live/default changes, exchange orders, deployment or merge follow this study.
