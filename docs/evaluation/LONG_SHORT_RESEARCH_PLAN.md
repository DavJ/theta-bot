# Fixed long/short futures experiment

Recorded 2026-10-06 before any economic replay in this round. The existing
50%-capped momentum spot account (+103.91% over Jan 2022-Sep 2026, CAGR 16.18%)
does not meet the user's profitability ambition. Preserve the **20% historical
research drawdown budget**. All periods and the three surviving assets were
already inspected; this is research, not a fresh market lockbox.

## Nine candidates and fixed sizing

1. `spot_control`: unchanged M30 positive/inverse-volatility, aggregate 50%,
   individual 25%, Monday rebalance and daily inactive/cap exits.
2. `perp_long_control`: identical completed-spot-close targets and schedule,
   including original individual cap clipping, but real perpetual execution/mark
   prices and funding. Isolates instrument.
3. `signed_m30`: sign of 30-day spot return, inverse 60-day daily volatility.
4. `signed_horizons`: average signs of 7/30/90-day returns, same inverse-vol
   normalization across all three assets; require all horizons complete.
5. `ema`: sign of EMA12 minus EMA48, require 48 observations; inverse vol.
6. `breakout`: symmetric 20-day entry /10-day exit close channel, all channel
   bounds from preceding completed closes. Retain direction until exit/flip.
7. `relative`: long strongest M30 asset, short weakest, stable column-order
   tie handling; equal absolute dollar weights, no trade when spread is zero.
8. `pairs`: ETH/BTC and BNB/BTC log-price regression, preceding 90 closes only.
   Beta clipped to [0.25,3], z entry +/-2, exit |z|<=0.5 or sign crossing,
   forced flat at |z|>=4. Hedge dollar ratio beta; two pairs equally combined.
9. `blend`: equal average allocations from signed_horizons, EMA, relative,
   and pairs. No fitted allocation meta-model in this round.

New strategies target <=100% gross notional and <=50% absolute per asset;
uniform scaling preserves hedge ratios. Use the fixed annual volatility target
20%, trailing 60 completed-close covariance, 25% diagonal shrinkage, scale down
only. This is **not** a guaranteed 20% drawdown controller. Daily ordinary
rebalances, 1% NAV band, 10 USDT minimum; exits and gross/asset cap reductions
bypass the band. Quantities use known open NAV, never current bar high/low or
future funding. Do not search parameters or add variants after economic results.

## Data, funding and ledger

Use existing checksum-verified spot daily signals, actual USD-M perpetual daily
trade OHLCV for fills, hourly mark OHLC for NAV/bounds, and all original settled
funding events (timestamps including jitter and changing settlement intervals).
Verify every official monthly SHA-256 checksum, unique grid, value bounds and
hashes of assembled files. Require complete Jan 2022-Sep 2026 coverage; do not
replace missing futures prices with spot prices or fabricate funding rates.

Pre-replay source correction: BTC hourly mark archive July 2022 omits July 31.
Missing monthly bars may be repaired only from the matching official daily
archive with its own checksum, identical overlapping observations, and a final
complete grid. Record all such supplements; never interpolate prices.

Older Binance funding API observations have no markPrice. Conservatively bound
each actual funding payment with the corresponding hourly mark high/low: charge
positive signed quantity*rate at the hourly high; value negative payment (credit)
at the hourly low. At an execution hour, use the larger payment/smaller credit
of the before/after quantities, including zero. Binance does not guarantee
the exact funding settlement instant around the timestamp. Cash expense is
resolved after the open decision; the realized rate cannot affect same-open
sizing. This is a conservative funding approximation, not exact tick settlement.

Futures collateral cash plus unrealized P&L is NAV. Signed position increases
update average entry, reductions realize only reduced quantity, and flips close
old inventory before opening the opposite direction. Fees debit cash once;
spread/slippage enter fill price and P&L once. Funding transfers debit/credit
cash once. Hourly mark bounds use high for long highs and short lows, low for
long lows and short highs, with a peak that never resets. A 5%-of-gross hourly
maintenance screen is a conservative research assumption, not an exchange
margin schedule or a simulated guaranteed liquidation fill. Any breach excludes
the candidate; negative collateral cash also excludes it (no unmodeled credit).
Actual exchange liquidation/ADL/insurance behavior is not claimed.

## Selection and later checks

Initial capital 1,000 USDT; fee 0.1% per fill, slippage 5bps, full spread 2bps.
Stress doubles fee and slippage; spread unchanged, actual funding unchanged.
Freeze highest positive net-return development candidate on a single 2022-2024
account with conservative hourly/daily drawdown bound <=20%, positive NAV and
no maintenance breach. Write this choice before later economic metrics.

Report all nine on 2025, Jan-Sep 2026, one continuous 2025-Sep 2026 account,
and one continuous 2022-Sep 2026 account, each at ordinary and doubled costs.
Do not reselect on later results. Require the frozen choice to improve the
spot control's net return on both continuous later and full paths, remain
positive in each separately started newer year and all cost stresses, and stay
within the 20% bound in every path. Different daily/hourly observation grain
is disclosed; the futures long-only control isolates that difference.

Reconcile cash/P&L/funding/fees/NAV, test long and short gains/losses, partial
reductions and flips, preserved funding jitter/signs, missing data rejection,
future-prefix invariance, and no same-bar mark/funding input to targets.
No live orders, deployment, leverage setting changes or merge.

Sources: [Binance public archives](https://github.com/binance/binance-public-data),
[funding calculation and settlement timing](https://www.binance.com/en/support/faq/detail/360033525031),
[crypto momentum research](https://www.nber.org/papers/w24877).
