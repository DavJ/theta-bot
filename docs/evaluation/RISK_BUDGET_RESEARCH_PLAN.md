# Fixed 20% drawdown research protocol

Recorded on 2026-10-06 before running this comparison. The user increased the
acceptable account drawdown from 10% to **20%**. This is a selection constraint,
not a guarantee that an execution rule can cap future losses.

The previously explored 30-day positive-momentum/inverse-volatility allocation
on BTC/ETH/BNB remains fixed. Do not tune its signals or add candidates after
inspecting this experiment. All historical periods and this survivor universe
are already known; later periods are chronological diagnostics, not fresh data.

## Seven fixed allocation variants

All use spot cash only, aggregate target cap at most 100%, individual cap equal
to half the aggregate cap, Monday rebalancing, daily inactive/risk reductions,
1% NAV ordinary rebalance band, 10 USDT minimum trade and the existing market
execution accounting. No borrowing, shorts or live orders.

| Variant | Aggregate cap | Annual volatility target | Cushion multiplier |
|---|---:|---:|---:|
| static_50 | 50% | none | none |
| static_75 | 75% | none | none |
| vol_15 | 100% | 15% | none |
| vol_25 | 100% | 25% | none |
| cushion_3 | 100% | none | 3 |
| cushion_5 | 100% | none | 5 |
| vol_25_cushion_5 | 100% | 25% | 5 |

Volatility uses the last 60 completed daily close-to-close returns, annualized
with 365 days, and a covariance matrix shrunk 25% toward its diagonal. Current
portfolio correlations are included. A volatility-limited policy stays in cash
until its covariance is ready. It only scales exposure down, never uses leverage.

The cushion rule retains a 2 percentage point reserve inside the 20% budget:
floor = 82% of the conservative historical account peak. Permitted risky
fraction = min(100%, multiplier * max(0, NAV - floor) / NAV). No peak resets or
automatic new budget after losses. Hitting the floor can leave the account in
cash indefinitely. Reductions are checked at each known daily open, including
non-rebalance days, and bypass the ordinary rebalance band. Minimum trade size
still applies. Current daily highs/lows are never used for current order sizing.

The risk peak includes previous completed-day portfolio high bounds. Holdings
are fixed between daily open trades and daily close, so summing each asset's
high supplies an upper bound, and summing lows a lower bound. These extrema
need not occur together or high before low: the resulting drawdown bound is
deliberately conservative, not a measured intraday portfolio drawdown. Include
pre- and post-fill open NAV, costs and previous peaks. Keep conventional
close-only drawdown and sampled open/close drawdown separately for inspection.

## Selection and diagnostics

1. Run one continuous 1,000 USDT development account from 2022-01-01 through
   2024-12-31. The first 60 days warm the model; no pre-2022 data are available.
2. Freeze the highest positive net-return variant whose conservative drawdown
   bound is at most 20%. Otherwise freeze cash. Write the selection before
   evaluating subsequent periods; do not switch to a later apparent winner.
3. Evaluate all seven on independently restarted 2025 and Jan-Sep 2026 accounts,
   and one continuous account from 2022 through Sep 2026 with no peak/cash reset.
4. Repeat the continuous account and both later periods with double fees and
   slippage: base fee 0.1% per fill, slippage 5 bps, spread 2 bps; stress fee
   0.2%, slippage 10 bps, same spread.
5. Fixed gap diagnostic: scale every asset's OHLC by 0.6 from 2024-03-11 onward,
   without changing volume or reconstructing that day's open. This adds a
   simultaneous 40% open gap and retains subsequent relative returns. Run the
   full continuous account. This artificial shock is a diagnostic, not a claim
   about its probability or an independent historical observation.
6. Historical screen for the frozen choice requires positive net returns in
   both later nominal and stressed periods, positive continuous stressed return,
   and conservative drawdown <=20% in all nominal and cost-stressed evaluations.
   Report the artificial gap result separately; a historical screen never
   establishes a guaranteed future loss bound or enables live execution.

Save equity/trade CSVs for reconciliation, summaries, complete source manifests,
the frozen choice, each screening gate, and the exact risk settings. Reconcile
fees/impact to executed trades, cash/base nonnegativity and portfolio NAV. Add
regression checks for prefix causality, risk reductions between rebalances,
minimum-notional residuals, unreplenished budgets and gap overshoot.

Conceptual references, not evidence that this crypto strategy is profitable:
[Moreira and Muir, Volatility Managed Portfolios](https://www.nber.org/papers/w22208)
and [SEC investor bulletin on order types and execution prices](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-14).
