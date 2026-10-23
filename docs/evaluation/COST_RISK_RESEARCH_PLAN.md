# Fixed cost/risk execution study — 2026-10-13

Written before the new policy payoffs. This study improves execution and checks
the risk of actual holdings; it does not add another return predictor.
It builds on f2500b04e2b0f52d1d8b6a8d92c770db2f046c26 (draft PR #129).

## Policies and fixed candidate

Keep the existing capped 30-day momentum targets, 50% aggregate/25% asset caps,
Monday ordinary rebalancing, 10-USDT minimum, funded spot inventory and cash.

1. `spot_control`: original market replay and 1-percentage-point trading band.
2. `band_edge`: ordinary rebalances stop at the nearest edge of the same band,
   instead of trading all the way to the target. Inactivity and cap sales bypass
   the band. Both allocation caps still apply after the boundary adjustment.
3. `vol_guard`: original rebalance destination plus a 15% annualized portfolio
   volatility ceiling, estimated with the existing 60-completed-day covariance
   and 25% diagonal shrinkage. Risk checks occur on every 4h open and use actual
   holdings between scheduled decisions; reductions bypass the ordinary band.
4. `guarded_edge`: both changes above. This is the **fixed candidate**, chosen
   before computing development payoffs. The other policies are ablations and
   cannot replace it after inspecting results.

All covariance matrices are published with the completed day's 20h bar label
and lagged by the same 1/2 bars as targets. Incomplete final days must not publish
their close. Before adequate history, the volatility guard requests cash.
The guard never increases the proposed allocation. There is no drawdown-reset,
loss multiplier, additional capital, leverage or option sleeve in this study.

## Frozen replay and acceptance

Use the same 171 checked official BTC/ETH/BNB spot archives, Jan 2022–Sep 2026,
31,212 bars. These dates were already studied; no fresh holdout is claimed.
Two independently funded initial accounts: 1,000 and 10,000 USDT.
Nominal per-fill costs: 0.1% fee, 5bps slippage, half of a 2bps spread.
Stress doubles fee/slippage; spread stays the same. Four scenarios: nominal,
cost stress, additional 4h signal/risk-information lag, both together.
Risk decisions still occur at every open on lagged information; the timing
scenario is not a claim to model delayed emergency-order execution.

Replay 2022–2024 development first; save its results and the fixed candidate
before replaying independently funded 2025, Jan–Sep 2026 and continuous later/
full accounts. Report all four policies and scenarios (160 account paths).
Peaks, inventories and losses persist on continuous accounts. Terminal owned
inventory is marked to market; exit costs are not silently invented.

The fixed candidate improves the historical profit screen only if, for both
account sizes, it earns more than control in every nominal later/year/full
comparison and in full joint stress, remains profitable under every later/full
stress scenario, and all five periods/four scenarios have conservative 4h
drawdown no worse than -20%. Account ledgers must independently reconcile.
Separately report a risk-improvement screen: the same accounting, positive
later/full stress results and 20% drawdown budget, plus lower worst full-period
drawdown than control. A risk-only pass is not a profitability improvement.

No parameter tuning, expanded grid, later winner selection or live eligibility.
The 20% historical budget and estimated volatility are not future loss caps.
Market gaps, minimum-notional residuals, unavailable markets and execution costs
can exceed a requested risk target before an order can reduce a position.
No trading, deployment, live defaults or PR merge is authorized.

## Evidence and checks

Persist the protocol hash and pre-payoff timestamp, development selection,
source manifest, runtime versions, complete metrics and checksummed CSV ledgers.
Retain full-period ledgers for every policy/scenario/account. Verify cash,
inventory, NAV, costs and completed information independently. Reproduce all
40 original control paths against the archived Bayesian study, including the
original development paths. Test covariance availability/prefix invariance,
off-switch equivalence, band/cap exits, risk reductions outside Monday, future
high/low isolation, minimum-notional residuals and invalid covariance inputs.
Do not interpret passing code tests as financial validation.

Prior evidence: [Bayesian study](THETABOT_BAYES_MONTE_CARLO_2026-10-12.md),
[old daily risk study](THETABOT_RISK_BUDGET_2026-10-06.md),
[spot policy](SPOT_ONLY_POLICY.md).
