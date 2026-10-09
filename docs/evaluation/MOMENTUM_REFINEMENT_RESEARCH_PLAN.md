# Fixed momentum-refinement experiment — 2026-10-09

Written before computing any new variant's economic results. This round changes
the profitable completed-day 30-day momentum policy rather than replacing it
with another fast predictor. Follow [the spot-only policy](SPOT_ONLY_POLICY.md):
owned cash/inventory only, no leverage, loans, margin, shorts, derivatives,
leveraged tokens or martingale. Live orders, defaults and deployment are outside
this experiment. The implementation depends on the spot guards in PR #121.

## Data, information clock and unchanged execution

Reuse the 171 checksum-verified official Binance BTCUSDT/ETHUSDT/BNBUSDT spot
monthly 4h archives, January 2022–September 2026. Require identical complete
UTC grids and validated OHLCV/taker columns; reject gaps and bad provenance.
Aggregate complete calendar days for signals. Publish the daily target only
with the last completed 4h candle of that day and execute at the next open.
No external, funding, news or order-book feature is added in this round.

One shared 1,000-USDT account, selling before buying. Ordinary rebalancing is
Monday at 00:00 UTC; inactive exits and cap reductions can occur at any daily
decision. All variants keep 50% aggregate/25% per-asset target caps, a 10-USDT
minimum fill and nonnegative cash/inventory including fees. Between decisions,
price moves may cause exposure to drift above target caps. End-of-period
inventory is marked to market, not liquidated. Drawdown peaks never reset.

## Seven fixed variants

Let M30 be the completed-close 30-day simple return and V60 the population
standard deviation of the previous 60 daily returns, including the completed
current return, floored at 0.0001. No exposure before V60 is ready. The control
normalizes positive-M30 assets' inverse V60 scores to 50%, then clips each
asset at 25% without redistributing excess weight.

| Variant | Fixed difference from the control |
|---|---|
| `spot_control` | Existing positive M30 / inverse V60; 1 percentage-point ordinary rebalance band. |
| `wide_band` | Same targets; 3 percentage-point ordinary rebalance band. |
| `confirmed_entry` | Entry requires three consecutive completed positive-M30 days; after entry stay eligible until M30 <= 0. The entry state describes signal eligibility, not an actual filled position. |
| `strong_entry` | Entry requires M30 > 0.25 × V60 × sqrt(30); stay eligible until M30 <= 0. This is a strength heuristic, not a significance test or forecast of net future returns. |
| `strong_entry_wide` | Same strength entry plus the fixed 3 percentage-point rebalance band. |
| `btc_regime` | ETH and BNB additionally require BTC M30 > 0 each completed day; BTC uses its own positive M30. |
| `capped_allocation` | Same eligible assets as control, but redistribute clipped inverse-volatility weight among eligible assets until the 50% aggregate budget or their 25% caps are reached. One eligible asset still receives at most 25%. |

Inactive exits and cap reductions bypass both rebalance bands. No negative-M30
holding is preserved to save costs. No parameter grid, additional post-result
variant, loss-based multiplier or later-period tuning is permitted in this
round. The broader band aims to reduce small ordinary adjustments; it cannot
be assumed to reduce total costs because future holdings and cap sales change.
The redistribution variant can use more of the existing permitted cash budget;
it does not raise the target caps, and must independently pass the drawdown test.

## Selection, costs and comparisons

First replay continuous 2022–2024 development accounts for all seven variants.
Freeze the largest positive net profit whose conservative 4h high/low drawdown
bound is no worse than -20%; otherwise freeze cash. Include the control in
selection. Write the protocol SHA-256 and frozen choice to disk BEFORE any
later-period economic replay. If the control wins, that is not an improvement.

Disclose all variants on fresh 2025 and January–September 2026 accounts,
continuous January 2025–September 2026 and continuous full history. Nominal
fills pay 0.1% fee +5bps slippage +half a 2bps spread per side. Repeat each later
path with 0.2% fee +10bps slippage and the unchanged spread. Keep signal rules
fixed under cost stress; impact is included in fill prices, not debited twice.
Stress the frozen choice AND control with an additional 4h delay, moving their
daily decisions to 04:00 UTC, on all four later/full periods at nominal costs.

A historical improvement requires the development-frozen NEW variant to beat
control on 2025, 2026, the continuous later account AND the continuous full
account; retain positive continuous later profit under doubled costs; stay
within the 20% bound on nominal and doubled-cost full accounts; and pass all
independent cash/inventory reconciliation. Report the timing comparison without
using it to select a different candidate. Later winners cannot replace the
development-frozen choice.

Report net profit, terminal equity, CAGR, conservative bound and close-only
drawdown separately; fills and active trading days; fills per week/month;
turnover, fee and impact totals; and full-history rolling 7/30/90-day endpoint
gains and doubling counts. Windows overlap and do not estimate independent
future probabilities. Independently reconcile every 4h close from signed fill
cash flows and inventory. Verify the control reproduces the preceding report
and the original daily engine; test causal prefix invariance and that exits/
cap reductions bypass the wider ordinary band.

## Interpretation

These historical periods were already studied, and the three-asset universe
contains surviving assets. This is a fixed exploratory comparison, not a fresh
holdout or a claim of repeatable weekly/monthly doubling. A 20% historical
rejection threshold does not guarantee a maximum future loss. Cost assumptions
are modeled, not account-specific measured fills. The conservative high/low
bound permits unfavorable cross-asset/intracandle ordering, not exact tick
drawdown. A positive result would still need subsequent unseen data and paper
execution before any live decision; it does not authorize such a decision.

Transaction-cost motivation: [Gârleanu and Pedersen, Dynamic Trading with
Predictable Returns and Transaction Costs](https://www.nber.org/papers/w15205).
The paper motivates avoiding unnecessary adjustments; its empirical futures
study is not evidence for this spot policy or permission to use derivatives.
Archive documentation: https://github.com/binance/binance-public-data .
