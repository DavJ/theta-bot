# Spot-only constraint — 2026-10-07

The user's binding constraint is to trade owned funds only. Leverage, borrowing,
margin accounts, shorts, futures, options, leveraged tokens and martingale
loss-doubling are excluded from further development and strategy selection.
Historical derivative reports remain records of rejected research, not permitted
strategies. No live trading or deployment is authorized by an offline experiment.

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
