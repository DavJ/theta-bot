# Theta Bot: multiscale spot momentum — 2026-10-09

**No frozen new policy passes the fixed historical profit/cost/timing screen.**
Development-frozen choice: `capped_control`. Live eligibility: **false**.
Dates and the surviving three-asset universe were studied previously; results remain exploratory.
No live orders, default changes, deployment or PR merge are enabled.

## Fixed hypothesis and execution

Two controls plus six fixed changes: 14/30/60-day momentum agreement, dual 30/90-day
agreement, weekly top-two ranking by 30-day or median multihorizon standardized strength,
and an actual-buy restriction after a sharp seven-day rise on the capped/consensus policies.
Read [the protocol](MULTISCALE_SPOT_RESEARCH_PLAN.md) and [spot-only policy](SPOT_ONLY_POLICY.md).
Protocol SHA-256: `15b519c0a2adb7cd22e66e797a0c106aadea32a5d0fe37b72e08f79d964933b5`; recorded before new economic results.

The same 171 checked official BTC/ETH/BNB spot archives supply 31,212 complete 4h bars
(January 2022–September 2026). Daily targets and permissions are published only with the
completed day's final 4h candle. The engine shifts both to the following open. Weekly ranks
are selected at completed Sunday closes; lost momentum eligibility can exit daily, but a
new ranked member waits until the next weekly selection. Ties follow stable column order.

One funded 1,000-USDT account, 50% aggregate/25% asset target caps, Monday ordinary
rebalance, 1pt band and 10-USDT minimum fill. Daily inactive exits and cap sales bypass
the ordinary band and buy restriction. Forbidden increases remain cash. The permission
compares targets with actual owned inventory and also applies to additions after entry.
The optional buy mask defaults to unrestricted buys and preserves previous control results.
No leverage, borrowing, margin, financial shorts, derivatives or martingale is introduced.

The six hypotheses are fixed heuristics, not calibrated future net-profit forecasts or p-values.
No new grid, later-period selection or post-result variant is used. Continuous paths preserve
peaks, losses, costs and compounding. Inventory is marked to market at period ends.

Nominal per-fill costs: 0.1% fee +5bps slippage +half a 2bps spread. Stress doubles fees and
slippage, leaving spread unchanged. Impact is already in the fill price, not debited twice.
Every variant also receives the same extra 4h target/permission delay, moving decisions to
04:00 UTC. This hypothetical timing stress is not an estimate of measured latency.

## Development selection

Highest positive continuous 2022–2024 profit under a -20% conservative 4h drawdown bound,
including both controls. The frozen choice is persisted before later-period economic replay.

| Variant | Development net | Development DD bound | Eligible |
|---|---:|---:|:---:|
| `spot_control` | +54.33% | -17.86% | yes |
| `capped_control` | +54.81% | -18.00% | yes |
| `consensus` | +43.89% | -19.32% | yes |
| `dual_horizon` | +45.57% | -13.91% | yes |
| `ranked_m30` | +52.76% | -20.81% | no |
| `ranked_consensus` | +49.78% | -20.77% | no |
| `capped_no_chase` | +48.00% | -18.96% | yes |
| `consensus_no_chase` | +39.97% | -20.62% | no |

## Net account results

2025 and Jan–Sep 2026 begin independent accounts. Continuous columns carry all results
forward. These are fresh accounts on known dates, not new unseen holdout periods.

| Variant | 2025 net | Jan–Sep 2026 net | Continuous 2025–Sep 2026 | Full 2022–Sep 2026 | Full CAGR | Full DD bound | Full doubled-cost net |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | +19.59% | +10.18% | +31.75% | +103.91% | +16.19% | -18.54% | +88.24% |
| `capped_control` | +21.04% | +12.02% | +35.45% | +110.26% | +16.95% | -18.62% | +93.80% |
| `consensus` | +13.48% | +12.75% | +27.96% | +85.03% | +13.84% | -19.32% | +70.24% |
| `dual_horizon` | +15.39% | +0.59% | +15.96% | +69.65% | +11.78% | -14.53% | +60.03% |
| `ranked_m30` | +29.15% | +9.06% | +40.85% | +115.49% | +17.55% | -20.81% | +98.06% |
| `ranked_consensus` | +25.77% | +12.37% | +41.37% | +112.20% | +17.17% | -20.77% | +95.05% |
| `capped_no_chase` | +21.11% | +12.03% | +35.54% | +101.11% | +15.85% | -18.96% | +85.40% |
| `consensus_no_chase` | +16.11% | +12.76% | +31.00% | +84.15% | +13.73% | -20.62% | +69.49% |

The descriptive full-nominal leader is `ranked_m30`: terminal NAV 2154.86 USDT
from 1,000 USDT, close-only DD -20.35%, conservative high/low bound -20.81%.
This description does not replace the development-frozen choice or bypass any stress gate.
The high/low bound assumes unfavorable intracandle/cross-asset ordering, not exact tick
chronology. The 20% historical threshold does not guarantee a maximum future loss.

## Costs and trading frequency

One fill is one executed buy/sell leg, not a completed round trip. Full nominal accounts.

| Variant | Fills | Active days | Fills/month | Turnover / initial capital | Fees USDT | Impact USDT | Net P&L USDT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | 432 | 271 | 7.58 | 73.85 | 73.85 | 44.31 | +1039.10 |
| `capped_control` | 448 | 287 | 7.86 | 76.94 | 76.94 | 46.17 | +1102.65 |
| `consensus` | 435 | 281 | 7.63 | 74.25 | 74.25 | 44.55 | +850.27 |
| `dual_horizon` | 299 | 198 | 5.25 | 49.20 | 49.20 | 29.52 | +696.45 |
| `ranked_m30` | 361 | 292 | 6.33 | 79.71 | 79.71 | 47.83 | +1154.86 |
| `ranked_consensus` | 369 | 293 | 6.47 | 80.99 | 80.99 | 48.59 | +1122.00 |
| `capped_no_chase` | 416 | 279 | 7.30 | 73.26 | 73.26 | 43.95 | +1011.05 |
| `consensus_no_chase` | 427 | 279 | 7.49 | 71.86 | 71.86 | 43.12 | +841.50 |

All 104 paths independently reconcile cash, inventory and NAV at every 4h close.
Largest NAV residual: 2.96e-12 USDT. Sixteen nominal/stress checks reproduce both controls'
previous profits, drawdown, costs and fill counts within 1e-8 absolute tolerance. Minimum cash
and inventory are nonnegative. Cost addback is diagnostic, not a simulated no-cost account.

## Every-policy timing and cost-risk stress

No descriptive leader is exempt from the one-bar delay. A full delayed drawdown above 20%
fails this round's historical screen even when nominal profits are higher.

| Variant | Delayed 2025 net | Delayed 2026 net | Delayed later net | Delayed full net | Delayed full DD | Doubled-cost full DD |
|---|---:|---:|---:|---:|---:|---:|
| `spot_control` | +15.83% | +8.19% | +25.24% | +94.29% | -20.20% | -19.24% |
| `capped_control` | +16.74% | +9.79% | +28.12% | +99.54% | -20.37% | -19.32% |
| `consensus` | +11.37% | +13.69% | +26.58% | +77.53% | -17.91% | -19.96% |
| `dual_horizon` | +10.86% | +1.34% | +12.42% | +67.53% | -14.90% | -15.03% |
| `ranked_m30` | +22.98% | +7.25% | +31.54% | +104.28% | -20.15% | -21.40% |
| `ranked_consensus` | +22.42% | +12.95% | +38.05% | +103.45% | -18.23% | -21.37% |
| `capped_no_chase` | +17.32% | +9.60% | +28.53% | +89.35% | -20.37% | -19.55% |
| `consensus_no_chase` | +14.65% | +13.48% | +29.98% | +76.23% | -18.08% | -21.25% |

## Rolling short-horizon gains

Full nominal continuous account, calendar-day closing endpoints including initial capital.
Windows overlap and are historical observations, not independent future probabilities.

| Variant | 7d doublings/windows | Best 7d | 30d doublings/windows | Best 30d | 90d doublings/windows | Best 90d |
|---|---:|---:|---:|---:|---:|---:|
| `spot_control` | 0/1728 | +10.39% | 0/1705 | +27.02% | 0/1645 | +29.50% |
| `capped_control` | 0/1728 | +11.58% | 0/1705 | +27.19% | 0/1645 | +29.74% |
| `consensus` | 0/1728 | +11.58% | 0/1705 | +28.00% | 0/1645 | +29.65% |
| `dual_horizon` | 0/1728 | +9.96% | 0/1705 | +27.53% | 0/1645 | +30.02% |
| `ranked_m30` | 0/1728 | +11.53% | 0/1705 | +21.21% | 0/1645 | +31.89% |
| `ranked_consensus` | 0/1728 | +11.53% | 0/1705 | +22.90% | 0/1645 | +29.35% |
| `capped_no_chase` | 0/1728 | +11.58% | 0/1705 | +24.31% | 0/1645 | +26.12% |
| `consensus_no_chase` | 0/1728 | +11.58% | 0/1705 | +27.17% | 0/1645 | +28.06% |

## Frozen improvement gates

| Gate | Passed |
|---|:---:|
| development_eligible | true |
| new_candidate | false |
| accounting | true |
| controls_unchanged | true |
| beats_both_controls_2025 | false |
| beats_both_controls_2026 | false |
| beats_both_controls_later_continuous | false |
| beats_both_controls_full_continuous | false |
| full_nominal_drawdown | true |
| full_doubled_drawdown | true |
| full_delayed_drawdown | false |
| later_doubled_positive | true |
| later_delayed_positive | true |

Historical screen: **False**. No frozen new policy passes the fixed historical profit/cost/timing screen.
Selecting a control again cannot count as a new improvement. Passing code tests does not
prove profit, and even a passed historical screen on reused data would need unseen/paper confirmation.

## Reproduction and limitations

```bash
python -m scripts.download_spot_flow
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_multiscale_spot
python -m scripts.write_multiscale_spot_report
python -m pytest -q tests
```

Checked archives/manifest, frozen selection, completed targets/buy permissions and every
equity/fill CSV are reproducible under `data/raw/spot_flow/` and `artifacts/multiscale_spot_20261009/`
(ignored by Git). [THETABOT_MULTISCALE_SPOT_2026-10-09.json](THETABOT_MULTISCALE_SPOT_2026-10-09.json) retains all summaries, source URLs/hashes, signal
hashes, control checks and gates.
Evidence SHA-256: `5e0d16df64ef2285f1998b6a551b352548f2f1a681ca9badd58765854c38c3a3`.

- Previously studied periods; not a new unseen holdout.
- Three surviving assets; universe survivorship not controlled.
- Momentum, ranks and no-chase threshold are fixed heuristics, not calibrated forecasts or p-values.
- Modeled fees/impact, not actual fills or account-specific fee tiers.
- Overlapping rolling windows do not estimate independent future doubling probabilities.
- Conservative within-bar high/low bound, not exact tick drawdown.
- 20% historical budget does not guarantee future maximum loss.

Primary motivation: [time-series cryptocurrency momentum](https://www.nber.org/papers/w24877)
and [cross-sectional cryptocurrency factors](https://www.nber.org/papers/w25882). Their portfolios
and results are not replicated here; their long-short strategies are excluded by the spot-only policy.
Data schema/checksums: https://github.com/binance/binance-public-data .
