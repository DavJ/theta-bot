# Theta Bot: spot-only 4h flow study — 2026-10-07

**No profitability improvement. All six new variants lose after costs.**
The development-frozen choice is `spot_control`; the historical improvement screen is **False**.
No live trading, defaults, deployment or exchange orders are enabled.

The binding constraint excludes leverage, loans, margin, shorts, derivatives, leveraged tokens and
martingale loss-doubling. The live executor verifies a spot market and free spot funds before
submission; unavailable information rejects an order. The planner forbids `allow_short=True`.
Recorded fills reject cash borrowing and sales exceeding owned inventory, including fees.
This fixes an old impossible-fill accounting path that could credit an oversized sale and then
clamp inventory to zero. Cash/inventory clamping now handles numerical residue only.

## Data and fixed design

171 official monthly spot archives provide 31,212 complete 4h bars
for BTCUSDT, ETHUSDT and BNBUSDT, 2022-01-01 through 2026-09-30. Each published SHA-256
was checked; timestamp-unit transitions, candle close times, OHLC and taker-volume fields were
validated. No missing bars were filled. These are aggregated executed taker trades, not historical
order-book depth, bid/ask imbalance or tick-level microstructure data.

Read [the protocol](SPOT_FLOW_RESEARCH_PLAN.md) and [the spot-only policy](SPOT_ONLY_POLICY.md).
Protocol SHA-256: `ba4a800b23dc102f6ea23f2f03078f61bd9213c2a293ab8298040b02230d9aa7`. Written before new economic results.
Each signal uses completed bars and trades at the next open. A shared 1,000-USDT account sells
before buying. All seven paths retain 50% aggregate/25% per-asset target caps, a 1% rebalance
band and a 10-USDT minimum fill. None sizes from a loss-recovery multiplier.

Nominal fills pay 0.1% fee +5bps slippage +half a 2bps spread on each side. Stress fills pay
0.2% fee +10bps slippage with the same spread. Impact below includes slippage and half-spread;
it is already in the fill price and is not debited twice. No account fee tier or measured fills
are assumed. Open inventory stays marked to market at period ends.

## Development choice, frozen before later replay

Choose the largest positive 2022–2024 net profit with conservative 4h drawdown no worse than −20%.
The selection file is written before any later path. All variants are disclosed below.

| Variant | Development net | Development DD bound | Eligible |
|---|---:|---:|:---:|
| `spot_control` | +54.33% | -17.86% | yes |
| `price_momentum` | -54.01% | -55.33% | no |
| `flow_momentum` | -52.10% | -53.53% | no |
| `price_breakout` | -25.51% | -32.08% | no |
| `flow_breakout` | -14.84% | -23.93% | no |
| `flow_reversion` | -4.82% | -9.90% | no |
| `flow_blend` | -25.13% | -25.85% | no |

## Net account results

2025 and 2026 columns start with independent 1,000-USDT accounts. Continuous paths preserve
all costs, losses and compounding. All dates have already been studied; they are not a fresh holdout.

| Variant | 2025 net | Jan–Sep 2026 net | Continuous 2025–Sep 2026 | Full 2022–Sep 2026 | Full CAGR | Full DD bound | Full doubled-cost net |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | +19.59% | +10.18% | +31.75% | +103.91% | +16.19% | -18.54% | +88.24% |
| `price_momentum` | -26.61% | -11.23% | -34.82% | -69.91% | -22.35% | -72.24% | -91.66% |
| `flow_momentum` | -23.24% | -16.75% | -36.06% | -69.26% | -22.00% | -70.88% | -91.29% |
| `price_breakout` | -21.86% | -15.21% | -33.70% | -50.59% | -13.80% | -53.78% | -83.21% |
| `flow_breakout` | -17.90% | -15.26% | -30.39% | -40.70% | -10.42% | -44.44% | -75.13% |
| `flow_reversion` | -3.34% | -2.13% | -5.40% | -9.95% | -2.18% | -14.15% | -15.25% |
| `flow_blend` | -14.74% | -11.13% | -24.18% | -43.21% | -11.23% | -45.14% | -71.93% |

Control equity reaches 2039.10 USDT from 1,000 USDT over 1,734 days.
Its sampled 4h-close drawdown is -17.66%; conservative 4h high/low bound is
-18.54%. These marks differ from daily-close drawdown.
The high/low bound combines unfavorable cross-asset extrema and allows the high to precede
the low within a candle; it is conservative rather than an exact tick-level drawdown.
The 20% budget is a historical rejection threshold, not a future loss guarantee.

## Costs and cash accounting

| Variant, full nominal account | Fills | Turnover / initial capital | Fees USDT | Impact USDT | Net P&L USDT |
|---|---:|---:|---:|---:|---:|
| `spot_control` | 432 | 73.85 | 73.85 | 44.31 | +1039.10 |
| `price_momentum` | 5546 | 517.78 | 517.78 | 310.67 | -699.14 |
| `flow_momentum` | 4243 | 482.00 | 482.00 | 289.20 | -692.59 |
| `price_breakout` | 4201 | 525.99 | 525.99 | 315.59 | -505.95 |
| `flow_breakout` | 3235 | 474.29 | 474.29 | 284.57 | -406.95 |
| `flow_reversion` | 167 | 39.86 | 39.86 | 23.91 | -99.54 |
| `flow_blend` | 5969 | 368.00 | 368.00 | 220.80 | -432.08 |

All 67 paths reconcile cash, inventory and equity at **every 4h close**
against an independent signed cash-flow ledger. Largest equity residual: 6.03e-12 USDT.
Every account retains nonnegative cash and inventory. The eight nominal/stress control
comparisons reproduce the daily engine's daily-close equity and fill quantities/prices/fees/impact
within 1e−8 absolute tolerance. This reproduces the earlier spot net profits on the same dates.

Cost addback is a diagnostic, not a simulated no-cost account: removing costs would change later
position sizes and compounding. Trading more frequently has not established an economic edge.

## Weekly and monthly doubling

Full nominal continuous account; windows use calendar-day closing NAV, including initial capital.
Counts refer to rolling endpoint-to-endpoint doubling, not an intraday spike. Windows overlap and
are historical observations, not independent estimates of future probability.

| Variant | 7d doublings / windows | Best 7d | 30d doublings / windows | Best 30d | 90d doublings / windows | Best 90d |
|---|---:|---:|---:|---:|---:|---:|
| `spot_control` | 0 / 1728 | +10.39% | 0 / 1705 | +27.02% | 0 / 1645 | +29.50% |
| `price_momentum` | 0 / 1728 | +10.16% | 0 / 1705 | +12.49% | 0 / 1645 | +13.66% |
| `flow_momentum` | 0 / 1728 | +6.57% | 0 / 1705 | +9.27% | 0 / 1645 | +7.46% |
| `price_breakout` | 0 / 1728 | +10.41% | 0 / 1705 | +13.81% | 0 / 1645 | +22.42% |
| `flow_breakout` | 0 / 1728 | +9.25% | 0 / 1705 | +15.27% | 0 / 1645 | +23.14% |
| `flow_reversion` | 0 / 1728 | +1.71% | 0 / 1705 | +2.22% | 0 / 1645 | +2.21% |
| `flow_blend` | 0 / 1728 | +5.02% | 0 / 1705 | +8.06% | 0 / 1645 | +9.78% |

## Timing stress and screen

One extra 4h bar delays both the frozen signal and its scheduled decision. The control's daily
decision therefore moves from midnight to 04:00 UTC; this does not accidentally become a
whole-day delay. This is a hypothetical timing stress, not measured network or exchange latency.

| Frozen candidate period | Extra 4h delay net | Extra 4h delay DD bound |
|---|---:|---:|
| 2025 | +15.83% | -13.50% |
| 2026 | +8.19% | -10.99% |
| later_continuous | +25.24% | -20.27% |
| full_continuous | +94.29% | -20.20% |

| Historical improvement gate | Passed |
|---|:---:|
| development_eligible | true |
| beats_control_2025 | false |
| beats_control_2026 | false |
| beats_control_later_continuous | false |
| full_nominal_drawdown | true |
| full_doubled_drawdown | true |
| later_doubled_positive | true |
| accounting | true |

The control cannot count as its own improvement. No new strategy passes the profit criteria.
The flow-breakout filter reduces losses relative to the price breakout but remains unprofitable;
the low-turnover reversion model also loses. This rejects these six fixed implementations, not
all possible uses of executed flow or finer market data.

A subsequent research direction is a cost-aware entry gate with a fixed holding interval and
training-only calibration: trade only when predicted gain clears round-trip costs and uncertainty.
It must compare matched price/flow models, preserve spot caps and be frozen before subsequent
unseen/paper data. That gate and richer order-book data are **not tested or implemented here**.

## Reproduction and limitations

```bash
python -m scripts.download_spot_flow
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_spot_flow
python -m scripts.write_spot_flow_report
python -m pytest -q tests
```

Raw ZIPs, checked source manifest, frozen selection and all equity/trade CSVs are written under
`data/raw/spot_flow/` and `artifacts/spot_flow_20261007/` (ignored by Git). The adjacent committed
[THETABOT_SPOT_FLOW_2026-10-07.json](THETABOT_SPOT_FLOW_2026-10-07.json) includes every summary, source URL/hash and gate.
Evidence JSON SHA-256: `25d2dd1f76ec2e653cf7b59157a19bc36352225a1bc910d7509a2dd51dd52d27`.

- Previously studied historical periods; not a fresh holdout.
- Three surviving assets; universe survivorship not controlled.
- Aggregated taker-flow bars, not historical order-book depth.
- Modeled costs, not measured live fills or account-specific fee tier.
- Overlapping doubling windows are not independent future probabilities.
- Conservative within-bar high/low bound, not exact tick drawdown.
- 20% historical budget is not a future loss guarantee.

Primary source documentation: [Binance archive schema and checksums](https://github.com/binance/binance-public-data)
and [spot market-data endpoints](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints).
