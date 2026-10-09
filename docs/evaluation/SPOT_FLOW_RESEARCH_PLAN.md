# Fixed spot-flow experiment — 2026-10-07

This protocol is written before any new candidate's economic results are
computed. It tests whether faster decisions and completed-bar taker flow improve
the existing spot candidate. It follows [the binding spot-only constraint](SPOT_ONLY_POLICY.md).

## Data and information clock

BTCUSDT, ETHUSDT and BNBUSDT spot 4h klines, January 2022–September 2026,
from Binance's official monthly archives with published SHA-256 verification.
Require identical complete UTC grids, valid OHLCV, nonnegative quote volume and
trade counts, and taker buys no greater than total volume. Reject missing bars;
do not interpolate. Retain both base and quote taker-buy volume. These are
aggregated executed trades, not historical depth or order-book imbalance.
Documentation: https://github.com/binance/binance-public-data and
https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints .

Every target is computed at a completed close and executed at the next 4h open.
Future OHLC only marks subsequent equity/drawdown. Open positions are marked to
market, not force-liquidated at period ends. A further one-bar signal delay for
the frozen winner is a timing stress, not a claim of measured exchange latency.

## Seven fixed variants

- `spot_control`: existing positive 30-day momentum/inverse 60-day volatility,
  Monday rebalance, daily midnight risk/inactivity exits, 50% aggregate/25% asset
  target caps. Derive daily bars from the same spot 4h data. Preserve its daily
  decisions; finer intraday marks can change the drawdown bound.
- `price_momentum`: positive 24h AND 7-day momentum; inverse trailing 14-day
  4h return volatility. Decisions every 4h.
- `flow_momentum`: same momentum, additionally standardized 12h net taker-buy
  pressure >0.5. Pressure is sum(2*taker-buy-base - volume)/sum(volume);
  standardize against the previous 30 days (minimum 14), excluding current bar.
- `price_breakout`: enter above the prior 48h close-channel high, exit below the
  prior 12h low. Inverse trailing 14-day volatility allocation.
- `flow_breakout`: identical breakout, but entry also requires pressure z>0.5.
  Exits do not wait for flow confirmation.
- `flow_reversion`: in a positive 7-day trend, enter after selling pressure z<−1.5
  AND log price is below its trailing 48h mean by 2 standard deviations. Exit at
  the mean, negative 7-day trend, or after six 4h bars (24h maximum signal hold).
- `flow_blend`: equal average of the three flow variants' nonnegative targets.

All new allocations use identical 50% aggregate/25% asset target caps and a 1%
absolute rebalance band. Inactive positions and cap reductions bypass that band.
Minimum fill notional is 10 USDT; sell before buy in one shared 1,000-USDT cash
account. No additional parameter grids, model tuning, loss-based sizing or
post-result variant additions in this round.

## Selection and evaluation

Use one continuous 2022–2024 development account. Freeze the largest positive
net return among candidates whose conservative 4h high/low drawdown bound is no
worse than −20%; otherwise freeze cash. Write that choice before later replays.
Report every variant, including rejected ones, on fresh 2025 and January–September
2026 accounts, continuous January 2025–September 2026, and continuous full history.
The latter keeps losses and compounding instead of resetting annually.

Each fill pays 0.1% fee +5bps slippage +half of a 2bps spread. Repeat every later
path with 0.2% fee +10bps slippage and unchanged spread. Disclose fill count,
turnover, fees, impact and net P&L. Reconcile final equity independently from
signed fill cash flows and inventory. Require nonnegative cash/inventory.

Report 7/30/90-calendar-day rolling account gains, maximum gain and number of
doublings. Windows overlap, so their frequencies are historical observations,
not independent estimates of future probability. Report full-history CAGR and
conservative drawdown, separately from close-only drawdown.

A historical improvement requires the development-frozen winner to beat the
control on both later fresh periods and the continuous later account, stay within
20% on the full nominal AND doubled-cost paths, retain positive doubled-cost
later P&L, and pass cash/inventory accounting. Past periods and the surviving
three-asset universe have already been studied; this is not a fresh holdout.
Passing does not authorize live deployment or prove repeatable weekly/monthly
doubling. Next confirmation would require subsequent unseen data and paper
execution with actual fee tier, precision and observed fills.
