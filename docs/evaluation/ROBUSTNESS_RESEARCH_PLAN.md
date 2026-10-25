# Fixed calendar and asset-omission audit — 2026-10-25

The entry and fixed-risk outcomes have already been observed. PR #134 reports
10,000-USDT full nominal net of 83.37%, 85.19%, 96.10% and 98.13% for the original
15% control and entry-hysteresis requests of 15%, 17.5% and 20%. The 17.5%
diagnostic outperforms 20% under combined stress. This follow-up tests dependence
on assets and calendar periods; it does not tune or replace those fixed policies.
The fixed candidate remains `entry_20`. All dates are previously studied.

## Fixed replay design

Keep exactly the four policies from the preceding risk-utilization study.
Compare the original three-asset universe and exactly three omissions:
`without_BTCUSDT`, `without_ETHUSDT`, `without_BNBUSDT`. The omitted asset has
zero targets from account inception. The retained assets use their unchanged
signal eligibility; there is no new ranking, forecast or conditional asset switch.
Reapply the existing 50% total/25% per-asset capped redistribution. With at most
two eligible assets this assigns 25% to each eligible asset; the remainder stays
cash. This reallocates eligible budget, rather than merely subtracting a past
asset's P&L. All three source markets and covariance dimensions remain present
with zero held quantity/target for the omitted one. No real asset is removed.

Fund 10,000 USDT independently for development (2022–2024), later continuous
(2025–September 2026) and full continuous (2022–September 2026). Run nominal
costs and combined doubled-fee/slippage plus one extra 4h signal/covariance lag.
Four policies × four universes × two scenarios × three periods = 96 paths.
Smaller accounts and separate pure-cost/pure-delay omission tests are not covered.

Keep completed-day clocks and one-bar execution lag, 60-return covariance with
25% diagonal shrinkage, missing-history cash, every-4h actual-holdings reductions,
Monday ordinary rebalancing, 1pp band-edge execution, 10-USDT minimum and all
existing cap/inactivity sales. Owned cash and inventory remain nonnegative.
There is no leverage, borrowing, financial short or loss-based sizing.

Publish plan, implementation, source/runtime/input identities and preceding
report/evidence identities before new payoffs. Save the exclusive development
selection before later/full replay. Reproduce all 24 matching three-asset
summaries from the preceding report. Preserve each omission and diagnostic;
neither the most profitable subset nor a different risk request replaces the
fixed candidate after results. Do not change existing frozen source files.

## Accounting, calendar and concentration diagnostics

All 96 accounts must reconcile. Independently audit complete full-period raw CSV
accounts for all four policies in the original universe and the fixed candidate
in all three omissions, in both scenarios: 14 accounts. Reconstruct source-open
fills, fees/impact, nonnegative balances, NAV, continuous conservative drawdown,
lagged covariance and configured-risk fields without rescaling the evidence.

For those 14 accounts compute complete calendar-month returns from actual
month-end account NAV, including initial cash for the first month. Calendar-year
returns compound these same months without account resets; mark 2026 as partial.
Report every month, worst/best month, all rolling 12/24-month returns, number and
fraction of negative windows, and longest month-end underwater run. Overlapping
windows are dependent; their fractions are not future loss probabilities.

Measure the share of positive log-return contributions from the best three
positive months. Also report the mathematical compounded return after excluding
the best three calendar months. These are ex-post concentration diagnostics,
not an executable timing strategy or replay with those months disabled.

Attribute each 4h change in account NAV by asset using previous owned quantity
times the gap from previous close to current open, plus post-fill quantity times
current close minus open, less that asset's fill fees and adverse price impact.
Every bar's summed asset contributions must equal account NAV change, and each
account's summed contributions must equal its reported terminal net P&L.
Retain gross holding contributions, fees, impact and net amounts separately.
These additive USDT contributions do not estimate a stand-alone asset strategy;
the omission replays, with shared cash and different risk scaling, do that stress.

## Fixed descriptive gates and interpretation

The omission screen requires accounting and original-control reproduction;
candidate conservative drawdown within the historical 20% selection budget in
every omission path; and positive candidate later/full net in all omissions and
both scenarios. No requirement is imposed that every rolling year be profitable;
publish negative windows rather than hiding them in a binary profit claim.
Calendar/concentration diagnostics do not select another candidate or threshold.

This is a diagnostic on reused history and three surviving assets, with possible
selection/survivorship bias, not confirmatory out-of-sample evidence. A passing
screen does not establish reliable future income. Annual-volatility requests and
historical drawdown budgets do not cap future losses. Model fills, fixed fees,
terminal marks, account/EEA execution, taxes, operating costs, custody and quote
FX retain their earlier limitations. No live/default changes, deployment or
merge are authorized. Export checksummed JSON/CSV and reconstructable archives.
