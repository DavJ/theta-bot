# Position-size sensitivity, 2026-10-06

The previous exploratory momentum candidate is now fixed. This experiment only
measures position sizing, not a new alpha model or a fresh holdout. All its
2023–2026 prices/results have already been inspected.

Keep BTCUSDT/ETHUSDT/BNBUSDT, positive 30-day momentum, inverse 60-day volatility,
Monday rebalancing, daily cash exits, 1% NAV rebalance band and 10-USDT minimum
notional unchanged. Replay exactly four aggregate target caps: 30%, 50%, 75%,
100%, with individual caps equal to half each aggregate cap. These are unlevered
spot allocations, never borrowed exposure.

Use fresh 1000 USDT for every independent period: 2022 bearish-period diagnostic
(first 60 days are indicator warmup), 2023–2024, 2025, January–September 2026.
Also replay 2025 and 2026 with doubled fee/slippage. Nominal fees remain 0.1%
per fill, slippage 5 bps plus half the full 2 bps spread. Do not multiply the
old returns; recompute actual trades, fee affordability and compounding.

Report every result including losses, average exposure and maximum drawdown.
Additionally replay one continuous account from January 2022 through September
2026 at all four fixed caps, to expose drawdown across calendar boundaries. This
supplement is specified before inspecting continuous-account results, after the
independent-period replays; it changes no signal, cap or selection criterion.
There is no selection winner, no presumption that a larger drawdown is accepted
by the user, no leverage/funding/liquidation modeling and no live authorization.
The prior 15% development drawdown limit remains an explicit reference assumption,
not a stated user preference. This experiment does not loosen it silently.
