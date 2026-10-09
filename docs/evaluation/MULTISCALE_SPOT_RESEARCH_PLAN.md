# Fixed multiscale spot-momentum experiment — 2026-10-09

Written before any new variant's economic results. Prior momentum-entry filters
failed the development-frozen screen; capped redistribution showed a small
descriptive improvement. This round tests horizon agreement, relative strength
and an actual execution-time restriction on buying after sharp recent rises.
Follow [the spot-only policy](SPOT_ONLY_POLICY.md): owned funds only, no leverage,
loans, margin, shorts, derivatives, leveraged tokens or martingale. No live
defaults, orders, deployment or PR merge are authorized by these simulations.

## Data and unchanged account mechanics

Reuse the 171 checked official Binance BTC/ETH/BNB spot monthly 4h archives,
January 2022–September 2026. Require identical complete UTC grids, valid OHLCV
and manifest SHA-256 hashes; do not interpolate or add new external data.
Aggregate daily candles; publish a daily signal only with its completed last
4h candle, then trade at the next open. No future close/high/low may size a fill.

One shared 1,000-USDT account, 50% aggregate/25% per-asset target caps, 10-USDT
minimum notional, 1 percentage-point ordinary rebalance band. Monday 00:00 UTC
ordinary rebalance; daily cap reductions and exits when the variant's own
eligibility becomes inactive. Between decisions holdings may drift above target
caps. Sell before buy; cash/inventory including fees cannot become negative.
Do not liquidate at period ends or reset peaks on continuous paths.

## Eight fixed variants

M(h) is the completed-day h-day simple close return. V60 is the population
standard deviation of the last 60 completed daily returns, floored at 0.0001.
No weight before V60 is ready; longer-horizon variants wait for their inputs.
Capped proportional allocation redistributes inverse-V60 weight until the 50%
aggregate budget or eligible assets' 25% caps are reached; unused budget is cash.

| Variant | Fixed rule |
|---|---|
| `spot_control` | Existing positive M(30), inverse V60, then clip 25% without redistributing. |
| `capped_control` | Previous `capped_allocation`: positive M(30), capped proportional inverse V60. |
| `consensus` | At least two of M(14), M(30), M(60) are positive; capped proportional inverse V60. |
| `dual_horizon` | M(30) AND M(90) are positive; capped proportional inverse V60. |
| `ranked_m30` | Select at most two positive-M(30) assets by M(30)/(V60*sqrt(30)); each receives 25%. |
| `ranked_consensus` | Require the consensus rule; select at most two by the median of M(h)/(V60*sqrt(h)), h=14,30,60; each receives 25%. |
| `capped_no_chase` | Capped control targets, but prevent an actual buy/increase when completed M(7) > 2*V60*sqrt(7). |
| `consensus_no_chase` | Consensus targets with the identical actual-buy restriction. |

Rank membership is chosen only at each completed Sunday close, for Monday's
rebalance, and retained through that week. An asset loses daily eligibility
immediately when its own momentum/consensus condition ceases, but replacements
wait for next Sunday's selection. Stable input column order resolves score ties.
Ranks and standardized returns are heuristics, not calibrated net forecasts or
significance tests. The no-chase rule compares completed prior-day values with
the actual owned position at execution; it cannot prevent a sell, inactivity
exit or cap reduction. Withheld buys remain cash, not reassigned to other assets.
It applies to both initial entry and additions to an existing position.

No extra grid, post-result variant, fitted parameter or later-period tuning in
this round. Horizons, rank count, buy threshold, costs and caps are fixed here.

## Freeze, costs, timing and rejection criteria

First run all eight continuous 2022–2024 development accounts at nominal costs.
Freeze the highest positive net return with conservative 4h high/low drawdown
no worse than -20%, including both controls; otherwise cash. Write choice and
protocol SHA-256 before any later-period economic replay. A control selected
again is not a new improvement; later winners cannot replace the frozen choice.

Report every variant on fresh 2025 and Jan–Sep 2026 accounts, continuous Jan
2025–Sep 2026 and continuous full history. Nominal per-fill costs: 0.1% fee,
5bps slippage, half a 2bps spread. Repeat all later paths with 0.2% fee and
10bps slippage, spread unchanged. Signal/permission rules remain unchanged.
Impact is in fill prices, not separately debited a second time.

Also run EVERY variant on ALL four periods with one additional 4h signal and
permission delay at nominal costs. Move daily decisions to 04:00 UTC. This is
hypothetical timing stress, not measured latency. Prior control timing stress
breached 20%, so do not omit that test for a descriptive later leader again.

A historical improvement requires a development-frozen NEW variant to beat
BOTH controls on all four nominal evaluation periods, retain positive later
profit under doubled costs AND timing delay, satisfy the 20% full continuous
bound under nominal, doubled-cost AND delayed scenarios, and reconcile cash,
inventory and NAV independently. Compare the new/control delayed outcomes
without selecting a different model. Even passing this screen is exploratory.

Disclose terminal NAV, net P&L/CAGR, 4h-close and conservative-bound drawdown,
fees, impact, fills/active days/fills per week and month, turnover and rolling
7/30/90-day closing-endpoint gains and doubling counts. Windows overlap and do
not estimate independent future doubling probabilities. Independently reconcile
each 4h close from signed fill cash flows and owned inventory. Reproduce both
controls against the preceding report for every nominal/stress period. Test
future-prefix invariance, weekly rank selection and actual-buy restrictions,
including delayed permissions, exit/cap bypass and invalid mask rejection.

## Scope and evidence limits

All periods were studied previously; none is a new unseen holdout. Three
surviving assets give universe selection bias. The high/low bound permits
unfavorable intracandle/cross-asset ordering rather than exact tick chronology.
Modeled fees/impact are not measured live fills or an account-specific tier.
A 20% historical rejection threshold is not a guaranteed future loss cap.
No claim of repeatable weekly/monthly doubling follows from this experiment;
subsequent unseen data and paper execution would be required before live use.

Primary research motivation: [Liu and Tsyvinski, Risks and Returns of
Cryptocurrency](https://www.nber.org/papers/w24877) and [Liu, Tsyvinski and Wu,
Common Risk Factors in Cryptocurrency](https://www.nber.org/papers/w25882).
Their momentum findings motivate the hypotheses, not this fixed implementation
or its future profits. Their long-short portfolios are not implemented here.
Archive documentation: https://github.com/binance/binance-public-data .
