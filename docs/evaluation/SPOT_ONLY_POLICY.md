# Spot-only constraint — 2026-10-07

The user's binding constraint is to trade owned funds only. Leverage, borrowing,
margin accounts, shorts, futures, options, leveraged tokens and martingale
loss-doubling are excluded from further development and strategy selection.
Historical derivative reports remain records of rejected research, not permitted
strategies. No live trading or deployment is authorized by an offline experiment.

## Narrow research exception — 2026-10-10

The user subsequently authorized profitability modeling of bought (long) options,
with a future on/off control. Fully paid call/put research is now permitted in an
isolated cash sleeve. It is disabled by default and has no order adapter. Written
options, loans, leveraged/margin-funded positions and standalone futures remain
excluded. An exchange's instantaneous expiry settlement intermediate is not an
actively opened futures strategy. Economic leverage remains present in long options.
Limit premium plus fees, reserve the option capital from the original funded
account, and do not replenish option losses from spot. Historical trade prices
without bid/ask depth are price proxies, not executable quotes. Model valuations
are not actual historical option marks. This exception does not enable live trading.

The spot planner rejects `allow_short=True`. Simulated and recorded fills must
preserve nonnegative quote cash and base inventory, including fees. Live order
submission must verify a spot market and sufficient free spot balances; unknown
market/account information must reject the order rather than assume permission.

New allocations must have nonnegative weights, aggregate target exposure at most
50% and per-asset target at most 25% in this experiment. Position size does not
increase in response to realized losses. Signals may buy again after a losing
trade, but there is no loss-based doubling or recovery multiplier.

The 20% drawdown budget is a historical rejection criterion, not a guaranteed
future maximum loss. Spot prices, exchange custody and quote-currency value can
still fall; spot-only trading eliminates borrowed exposure, not market risk.
