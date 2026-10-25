# Public depth and paper-execution calibration — 2026-10-23

The next operational question is whether the assumed trading costs resemble
currently displayed spot liquidity. Freeze this plan, code and runtime before
reading new order books. This is an execution-cost diagnostic; its immediate
buy/sell probes are not strategy trades or a profitable trading strategy.
The earlier prospective study remains a separate, unchanged candle-price replay.

## Fixed scope

Observe BTCUSDT, ETHUSDT, BNBUSDT and, separately, BTCUSDC, ETHUSDC, BNBUSDC.
USDT is the old research quote. Public USDC availability is not a validation of
ported signals, account access or EEA execution. Quotes and P&L in different
assets are never pooled or treated as identical dollars. No derivatives, account
keys, real orders, balance lookup, transfers, background process or live adapter.

Collect public exchangeInfo and the top 100 bid/ask levels, with exact decimal
strings, source request parameters, local start/receipt times and elapsed time.
One invocation records one bounded batch. Preserve every raw response, including
errors, with checksums and its frozen-protocol reference. Imported connector
captures and the direct public-data-only GET collector use the same envelope.
Never substitute candles or synthetic quotes if the source is unavailable.

Validate public TRADING/spot/MARKET identity, ordered noncrossed positive depth,
price ticks and both LOT_SIZE and MARKET_LOT_SIZE. A zero market step disables
that constraint, not the ordinary lot constraint. Combine nonzero steps by their
exact decimal least common multiple and round quantities down. Preserve at least
the existing 10-quote modeled minimum notional. Public MIN_NOTIONAL/NOTIONAL
flags bound the simulated VWAP, but this is not the exchange's average/reference
price test. Execution-price rules, account limits and actual commission assets
remain unverified; report exchange_filters_fully_verified=false in all cases.

## Conditional probes

Independently fund each probe with 10,000 units of its own quote currency and no
base inventory. Fixed all-in buy budgets: 25, 100, 1,000 and 2,500 quote units.
Cap the budget at 25% of owned cash. Buy only visible asks, then sell only the
acquired quantity into visible bids of that same snapshot. This is a static
round-trip cost measurement, not elapsed-time trading or realized strategy P&L.
Fee model: quote-denominated 0.1% per leg, rounded up to public quote commission
precision; stress: 0.2% plus a hypothetical 5bps adverse price reserve per leg.
Actual account commissions, base/BNB fee payment, promotions and rebates are not
known. Do not report these modeled fees as fees observed on the user's account.

Walk each level and account for actual visible depth. Reject unsupported sizes,
insufficient visible depth, modeled notional violations, borrowed cash or short
inventory. Require the whole requested quantity for the probe; this conservative
FOK-style simulation does not replicate an exchange MARKET order's partial-fill
or expiry behavior. Each size/scenario reuses the snapshot as an independent
counterfactual; they do not consume a single shared liquidity pool or capital.

Report full spread, raw depth impact, both VWAPs, fees, filled quantities,
residual budget, funded cash/inventory and round-trip cost. Preserve rejections.
Use deterministic independent arithmetic audit of every accepted probe.

## Time quality and interpretation

Depth replies have lastUpdateId but no source event timestamp. A local receipt
time is not exchange book age, and an update ID must never be used as a clock.
Record source_book_age_ms=null and observed_executable_fills=0. Simulations
are conditional on those displayed prices/quantities still being available.
Reject a request taking more than 5,000ms, future/reversed receipt clocks,
metadata received more than one hour before the book, or clock/elapsed-time
disagreement exceeding 250ms. Keep rejected raw data rather than hiding it.
Offline replay uses the recorded receipt boundary, not a falsely refreshed time.

One batch cannot establish a cost distribution. The calibration coverage gate
requires at least 200 distinct successful batches per market spanning 30 days;
no strategy profitability or live gate can pass from these probes. Latency
reserve, fees, budgets and markets are not selected after observing spreads.
This change does not lower historical cost assumptions or alter trading defaults.

## Primary references

- Public depth payload and average-price distinction:
  https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market
- Public data-only host and supported GET endpoints:
  https://developers.binance.com/en/docs/products/spot/faqs/market_data_only
- Quantity and notional filters:
  https://github.com/binance/binance-spot-api-docs/blob/master/filters.md
- Reference-dependent execution limits:
  https://developers.binance.com/en/docs/products/spot/faqs/price_range_execution_rules

Keep implementation outside spot_bot and preserve the 2026-10-16 prospective
source fingerprints. A successful diagnostic never authorizes live deployment,
account trading or a merge. Full strategy-forward evidence is still required.
