# Fixed allocation benchmark audit — 2026-10-23

Does the current guarded momentum policy add net value over simple allocation
rules using the same execution machinery and risk settings? The fixed candidate
is the existing `guarded_edge`; this study adds no tuned signal or replacement.
All January 2022–September 2026 data and BTC/ETH/BNB assets were studied before.
This is a historical diagnostic, not an unseen holdout or prospective evidence.

## Fixed policies

- `guarded_edge`: unchanged positive 30-day momentum eligibility, capped inverse
  60-day daily volatility allocation, 50% total and 25% per-asset target limits.
- `cash`: zero allocation, zero assumed interest. USDT cash is not risk free.
- `equal_25`: 25% total target, one third of that budget per asset.
- `equal_50`: 50% total target, one third of that budget per asset.
- `inverse_vol_50`: the same capped inverse-60-day-volatility allocation as the
  candidate, but all three assets are eligible regardless of momentum.

The two equal budgets are fixed lower/full allocation references, not a search
for the exposure that wins on these dates. Inverse-volatility isolates removal
of the candidate's momentum eligibility rule. Equal allocations use only full
completed UTC days. Inverse volatility uses population standard deviation of
60 completed daily simple returns, floored at 0.0001, with the existing capped
redistribution. Publish at the completed day's last 4h label (20:00); the engine
executes only at a later open. Incomplete first/last days do not publish targets.

All policies use the existing Monday ordinary rebalancing, 1 percentage-point
band-edge destination, 10-USDT minimum trade, 50% total/25% asset target caps,
daily scheduled cap checks and every-4h actual-holdings volatility reductions.
The common guard requests 15% annualized volatility with 60 completed daily
returns and 25% covariance diagonal shrinkage; missing covariance requests cash.
These are guarded weekly allocation references, **not literal buy-and-hold**.
Same risk settings do not imply identical realized risk; disclose realized
daily volatility, mean exposure and conservative intraday drawdown for each.

## Fixed replays and criteria

Use the identical 171 checksum-verified official Binance monthly spot archives,
reader, shared-cash engine, accounting and fixed nominal/stress costs. Nominal:
fee 0.1% per fill, 5bps slippage plus half a 2bps spread. Cost stress: fee 0.2%,
10bps slippage and the same spread. Timing stress adds one 4h target/covariance
lag; combined stress includes both. No order-book or emergency-latency model.

Independently fund 1,000 and 10,000 USDT accounts for development 2022–2024,
later continuous 2025–September 2026, and full continuous 2022–September 2026.
Run all five policies in all four scenarios: 120 paths. Never add or splice
returns from separately funded periods. Preserve the unchanged candidate and
development results in an exclusive selection file before later/full payoffs.
Reproduce all 24 matching archived guarded-edge paths.

The historical incremental-value screen requires reconciled nonnegative funded
accounts; candidate drawdown bounds within 20% in every replay; positive
candidate net in every later/full stress; and candidate net above each of the
three invested references in later/full nominal and combined stress at both
capital sizes. No reference or candidate is selected after results. A failure
means the strong all-comparisons claim fails, even if some rows are favorable.
This screen does not establish statistical significance or exactly matched
risk-adjusted alpha. No post-hoc leverage or volatility matching is permitted.

For every invested reference, report candidate minus reference terminal NAV
and divide that difference by period length in 30.4375-day calendar months.
This is an undiscounted descriptive *incremental cost headroom*, not simulated
monthly withdrawals or an earnings forecast. Both policies can have operating
costs. Only additional candidate costs relative to that reference belong in
this comparison; taxes, labor, hosting and quote FX are not otherwise modeled.

## Evidence and scope

Freeze source, plan, runtime, settings and data provenance before the new payoffs;
publish this freeze before running economic comparisons. Export target CSVs,
all summaries and the full nominal/combined-stress 10,000-USDT cash, inventory
and fill CSVs for all five policies. Independently audit these ten accounts
against source opens, costs, lagged covariance and reported final economics.
Preserve checksummed, reconstructable evidence and all unfavorable results.

The fixed surviving universe creates selection limitations. Model prices are
not executable quotes; terminal inventory remains marked rather than forcibly
liquidated. USDT pairs do not establish EEA/account market eligibility. A USDC
or EUR version requires separate validation. No paper/live connection, default
change, deployment or merge follows. The 20% historical screen does not cap
future loss. Keep all new research outside spot_bot to preserve the existing
2026-10-16 prospective protocol's source fingerprints and eligibility status.
