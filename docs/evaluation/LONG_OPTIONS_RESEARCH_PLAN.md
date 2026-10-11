# Fully paid long options: fixed exploratory model — 2026-10-10

User authorization: "long opce by se mohly v theta botovi zapinat ci vypinan,
ted ale namodelujme ziskovost". This supersedes the earlier absolute option ban
only for optional, fully paid long-option research. Live defaults remain spot.

Freeze this file's SHA-256 before new payoff calculations. Reused spot history
is not an unseen holdout. All variants must be disclosed, including skipped buys.

## Capital and variants

- Compare independently funded 1,000 and 10,000 quote-unit accounts. Quote units
  are modeled USD equivalents; USDT/USDC/USD parity is an explicit assumption.
- `options_enabled=False` is the unchanged full-capital capped spot control.
- Enabled accounts partition initial capital: 90% same capped BTC/ETH/BNB spot
  policy, 10% isolated option cash. No transfers, borrowing or loss replenishment.
  A 90% spot / 10% idle-cash control identifies the cost of reserving capital.
- Four fixed signals: call-only ATM; call-only 5%-OTM; momentum-direction ATM;
  momentum-direction 5%-OTM. Call-only requires positive completed 30-day momentum;
  direction buys calls for positive and puts for negative momentum, never writes.
- Model each signal separately on inverse BTC/ETH options and linear USDC
  BTC/ETH options. Linear availability must follow actual contract creation dates;
  do not backdate the August 2025 launch.
- First Monday of each month 00:00 UTC; choose last-Friday-of-month 08:00 expiry.
  Determine strikes using only the preceding completed daily spot close. Choose
  the nearest listed eligible strike, ties lower strike, with creation <= entry.
  No replacement if the selected contract lacks a fresh trade or is unaffordable.
- Buy once per month, hold to expiry, no early-exit optimization. Max aggregate
  premium, trading fees and currency-conversion costs: 20% of current option cash
  (initially 2% of the full account). Split equally among eligible BTC/ETH signals;
  unspent allocations remain cash. Size down to the permitted contract lot.
- Inverse minimum size is max(recorded instrument minimum, current documented
  0.1 BTC / 1 ETH). Linear minimum is max(recorded minimum, 0.01 BTC / 0.1 ETH).
  These conservative floors do not reconstruct historical rule changes.

## Data and modeled costs

- Official Deribit history API expired instrument metadata, recent trade before
  each scheduled entry (at most four hours old), and official expiry delivery
  prices. Retain source URLs, parameters, hashes and selected trade IDs/timestamps.
- Last trade is not an ask quote. Nominal assumed buy premium = historical trade
  premium * 1.05, rounded up to the instrument tick. No claim of actual fills.
- Current standard fee scenario: option buy 0.03% underlying notional, capped at
  12.5% premium; expiry fee 0.015% notional, capped at 12.5% positive payoff. No
  delivery charge for zero payoff. This is a uniform scenario, not a dated fee audit.
- Inverse options are priced/settled in coin. Model immediate owned-cash purchase
  of the required coin and immediate sale of delivery proceeds using Binance spot
  prices at those 4h opens, 0.1% fee +5bps slippage +half a 2bps spread. No retained
  coin collateral, no portfolio margin. Linear conversion assumes stablecoin parity
  and includes 0.1% conversion fees entering/leaving the isolated USDC sleeve.
- Stress: entry markup 20%; doubled fees/conversion impact; all signals/entries
  delayed an extra 4h, using causal quotes available by that delayed entry.
  Also value held options at half/twice entry trade IV as a marking sensitivity.
- Expiry payoffs use the exchange delivery index, not Binance closing price.
- Between entries and expiry, mark options with a zero-rate Black-Scholes model
  and the entry trade's IV. Underlying uses actual spot OHLC. This is synthetic
  valuation only; reported intra-period drawdown is model dependent. Closed-month
  payoff profit does not depend on these synthetic option marks.
- Do not treat observed trade amounts as available book depth. Disclose missing
  trade coverage, lot skips, index/spot basis and actual fees versus modeled costs.

## Chronology and evidence

- Development 2022–2024; freeze highest positive total account net with modeled
  conservative 4h DD <=20%, including off and reserved-cash controls, before later
  economic replay. Independently freeze for both account sizes. A control chosen
  again is not a new profitable option strategy.
- Report 2025, Jan–Sep2026, continuous 2025–Sep2026 and full Jan2022–Sep2026.
  Carry all capital/costs/peaks in continuous accounts; no summing independent returns.
- Report total account net/CAGR, spot/option contributions, all fees/markup costs,
  option count/win rate/payoff multiples, net premium loss, monthly lot/quote skips,
  modeled DD and rolling 7/30/90-day account gains/doublings.
- Independently reconcile option cash flow/positions/NAV, verify nonnegative cash,
  exact off-switch equivalence, and reproduce prior spot control results.
- No live eligibility or proven executable profitability can be established from
  trade-price proxies and model marks, even if an exploratory profit screen passes.
- A 20% historical model DD is not a guaranteed future maximum loss. Venue and
  settlement-currency risks are not included in the payoff simulation.

## Primary sources

- https://statics.deribit.com/files/DeribitInstitutionalSetupGuide.pdf
- https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options
- https://support.deribit.com/hc/en-us/articles/31424932728093-Linear-USDC-Options
- https://support.deribit.com/hc/en-us/articles/25944746248989-Fees
- https://insights.deribit.com/education/usdc-settled-btc-eth-options-launch/
