# Fixed risk-utilization follow-up — 2026-10-25

The entry-hysteresis study has already been observed: at 10,000 USDT its full
nominal return improved from 83.37% to 85.19%, with slightly less turnover.
This is an explicitly adaptive follow-up on the same previously studied
January 2022–September 2026 BTC/ETH/BNB spot history, not a fresh holdout.
The request for greater profitability motivates one bounded comparison of
fixed risk requests. Higher exposure is not new predictive alpha.

## Fixed policies and execution

Compare exactly four policies:

- `control_15`: original positive-momentum guarded edge, 15% requested annual vol.
- `entry_15`: fixed 0.64% entry hysteresis, 15% requested annual vol.
- `entry_175`: the same entry hysteresis, 17.5% requested annual vol; diagnostic.
- `entry_20`: the same entry hysteresis, 20% requested annual vol; fixed candidate.

Do not replace a failing candidate with the intermediate diagnostic after results.
The only difference between entry policies is the constant volatility request.
Retain 50% total and 25% per-asset target caps, owned nonnegative cash/inventory,
60 completed daily returns and 25% diagonal covariance shrinkage, missing-history
cash, every-4h actual-inventory risk reductions, Monday ordinary rebalancing,
1pp band-edge execution, 10-USDT minimum, and all inactivity/cap sales.
There is no leverage, borrowing, shorting, loss-based sizing or martingale.
A 20% annual-volatility request is not a 20% maximum-drawdown guarantee.

Use the identical checksum-verified 171 monthly archives and input manifest
from the preceding study, including its source CSV identities. Publish this
plan, source/runtime/settings/input fingerprints and the preceding compressed
report identity before computing these new payoffs. Preserve earlier frozen
sources and protocols. Re-created entry-study outputs must be identified as
a reproduction, with their actual selection timestamp; do not backdate them.

Independently fund 1,000 and 10,000 USDT. Development is 2022–2024; later
continuous is 2025–September 2026; full continuous is 2022–September 2026.
Run nominal costs (0.1% fee, 5bps slippage, 2bps spread), doubled fee/slippage,
one extra 4h signal/covariance lag, and combined stress. Covariance and target
publication use completed daily 20h labels, lagged one bar by execution plus
the extra timing-stress bar. Four policies × two capitals × four scenarios ×
three periods give 96 paths. Save the fixed development selection exclusively
before later/full replay; do not splice independently funded period returns.

## Fixed descriptive gates and evidence

Require all accounts to reconcile; every candidate path's conservative continuous
drawdown bound to stay within the historical 20% selection budget; positive
candidate later/full net in all scenarios; and strictly greater candidate net
than both 15% controls for later/full nominal and combined stress at both capitals.
These reused-history gates are descriptive, not statistical confirmation.

Reproduce all 48 preceding 15% control summaries. Export all 96 summaries,
realized daily annual volatility, mean exposure, CAGR, fees/impact, fill counts
and drawdown. Preserve complete 10,000-USDT nominal/combined full ledgers for
all four policies and both signal targets. Independently reconstruct raw CSV
fills, nonnegative funds/inventory, NAV, price bounds, drawdown, completed/lagged
covariance, actual estimated volatility and unscheduled risk scaling at the
configured request. Do not rescale CSV risk fields to fit the old 15% auditor.
Export checksummed JSON and reconstructable evidence archive parts.

Modeled opens and fixed fees do not establish executable fills, queue/depth,
actual account commissions, EEA availability or account permissions. Taxes,
operating costs, custody and quote FX are excluded. Terminal inventory is
marked rather than liquidated. Three surviving assets imply universe bias.
No live/default changes, exchange orders, deployment or merge are authorized.
