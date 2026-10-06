# Execution and portfolio experiment, 2026-10-06

This protocol is written before obtaining ETH/BNB archives or inspecting the new
portfolio results. It does not claim that the previously inspected BTC periods
are new out-of-sample confirmation.

## Execution diagnosis

- Fix the one-way and round-trip cost definitions to match simulated fills.
- Enforce the documented hysteresis floor/cap as actual bounds.
- Correct completed-day signal availability: a kline timestamp labels its open;
  its close becomes available one candle interval later.
- Compare the existing limit/one-bar-close fallback with market execution at the
  next known open. Compare legacy theta, normalized theta, breakout and fusion.
- Fixed 2025 and January–September 2026 periods are diagnostic replays. No
  parameter search. Identical BTC data and 30% cap as the previous study.

## Fixed portfolio candidates

Universe: BTCUSDT, ETHUSDT, BNBUSDT, spot only. These instruments are chosen now;
their continued survival is known, so this is not a survivorship-free universe.
No futures, shorting, borrowing, or external account connection.

1. BTC-only close breakout: entry above previous 20 daily closes, exit below
   previous 10; 30% cap, market orders.
2. Three-asset same breakout, equal weight among active assets.
3. Three-asset positive 30-day momentum, inverse 60-day volatility weights.
4. Positive momentum at 30/90/180/365 days, mean of four binary long/cash
   signals; equal base allocation and conviction-scaled exposure.
5. Same four horizons, inverse 60-day volatility base allocation and conviction.
6. Positive 90-day momentum rotation: top two positive assets, equal weight.

Rebalance on Monday opens from completed daily closes. Exit an inactive asset
at the next daily open. Enforce the aggregate 30% and individual 15% portfolio
caps at daily decision prices; the BTC-only control has a 30% individual cap.
Ignore ordinary adjustments smaller than 1% of current NAV or 10 USDT.
Risk exits/cap reductions bypass that band, but still obey minimum notional.
Sell before buy using a shared cash account and the existing core fill/fee
accounting. Market fills pay fee 0.1%, slippage 5 bps and half a 2 bps spread
per side. Terminal positions are marked to market, not forcibly liquidated.

2022 provides warmup. Select on 2023–2024 by greatest positive net profit among
candidates with maximum drawdown no worse than 15%, otherwise cash. Write the
selection before viewing 2025/2026. Compare all candidates without reselecting.
2025 is historical validation with new ETH/BNB data; 2026 is cross-asset transfer
and diagnostic replay, not a future live test. Also independently reset each
2025 quarter and double fee/slippage for the frozen candidate.

Do not tune horizons, universe, schedule, caps or entry/exit thresholds after
observing these results. No live-eligibility claim is created by this study.
Report every candidate, fees, traded notional, drawdown and exposure, including
cash and an initial 30% equally split buy-and-hold capital reference.

Sources motivating the hypothesis (this is a spot adaptation, not replication):
- https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum
- https://www.nber.org/papers/w25882
- https://github.com/binance/binance-public-data
