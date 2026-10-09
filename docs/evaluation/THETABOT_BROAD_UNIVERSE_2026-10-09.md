# Theta Bot — fixed historical broad spot universe

**A material improvement is NOT supported by this experiment.** Development froze `broad_top5_vol20` before later account replay.
Historical improvement screen: **false**; material screen: **false**.
Live eligibility remains false. Owned spot only; no leverage, borrowing, margin, shorts,
derivatives, leveraged tokens or martingale. This study does not enable orders or change defaults.

## What changed and what was fixed

Expand from BTC/ETH/BNB to 15 underlying assets selected from the dated 2021-12-26 top-20
market-cap snapshot, with a verified December 2021 spot-USDT source requirement. Retain old
Terra's collapse and trading absence, distinguish it from new Terra 2.0, and follow the
documented 1:1 MATIC/POL identity migration. CRO's source is HTTP 404 and is not replaced.
Stablecoins and wrapped BTC are excluded. This fixed historical cohort is not the whole market.

Six new policies combine completed-day liquidity, momentum ranks, inverse volatility and
a reduction-only volatility budget. New assets need 90 contiguous complete days, mean quote
volume >=10m USDT/day, and historical top-ten cohort liquidity. Sunday ranks execute through
Monday rebalances; daily inactivity/cap reductions are allowed. Controls retain the old rules.

Shared 1,000-USDT accounts, 50% aggregate/25% asset target caps, 1pt rebalance band, 10-USDT
minimum fills and sell-before-buy execution. Targets published at the completed 20h candle
execute at the next 00h open; delay stress executes at 04h. Exposure can drift between decisions.
Nominal fills cost 0.1% fee +5bps slippage +half a 2bps spread; cost stress doubles fee/slippage.
Open inventory remains marked at period ends. Later/full accounts preserve compounding and peaks.

[Fixed protocol](BROAD_UNIVERSE_RESEARCH_PLAN.md), SHA-256 `de01c2c3cc5b92989d12f3ce3e99c793427f44476bcf6d9e444729aa0aa545b6`.
Frozen UTC timestamp: `2026-10-09 21:42:09.367916+00:00`; source-manifest SHA-256 `642f86f457016fa01aecccf0da0c506e0174632f690b5ac67fa88b9a325b7567`.
Its documented source-validation amendment was written before economic replay and changed no
strategy or economic threshold. Periods were already used in previous rounds; this is not an unseen holdout.

## Development selection: 2022–2024

Positive net, conservative 4h DD within 20% and zero held-unquoted bars; then highest net.

| Policy | Net | Conservative DD | Held-unquoted asset/bars | Eligible | Frozen |
|---|---:|---:|---:|:---:|:---:|
| `spot_control` | +54.33% | -17.86% | 0 | true | false |
| `capped_control` | +54.81% | -18.00% | 0 | true | false |
| `broad_inverse_vol` | +51.60% | -24.87% | 0 | false | false |
| `broad_top3` | +82.05% | -26.70% | 0 | false | false |
| `broad_top3_vol` | +96.38% | -26.13% | 0 | false | false |
| `broad_top5_vol` | +46.94% | -24.58% | 0 | false | false |
| `broad_top5_consensus` | +36.36% | -26.37% | 0 | false | false |
| `broad_top5_vol20` | +55.09% | -18.34% | 0 | true | true |

## Net results after modeled costs

2025 and Jan–Sep 2026 each start a fresh account. Later is one continuous Jan 2025–Sep 2026
account; full is one continuous Jan 2022–Sep 2026 account, not a sum of reset periods.

| Policy | 2025 net | Jan–Sep 2026 net | Later net | Full net | Full CAGR | Full close DD | Full conservative DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | +19.59% | +10.18% | +31.75% | +103.91% | +16.19% | -17.66% | -18.54% |
| `capped_control` | +21.04% | +12.02% | +35.45% | +110.26% | +16.95% | -17.72% | -18.62% |
| `broad_inverse_vol` | -8.78% | +9.40% | -0.22% | +53.20% | +9.40% | -24.62% | -25.19% |
| `broad_top3` | -4.80% | -8.93% | -13.29% | +58.65% | +10.21% | -35.18% | -35.94% |
| `broad_top3_vol` | +2.10% | -5.50% | -3.51% | +90.63% | +14.56% | -30.29% | -30.83% |
| `broad_top5_vol` | -1.54% | +0.56% | -1.16% | +45.97% | +8.29% | -25.62% | -26.54% |
| `broad_top5_consensus` | -13.21% | +8.34% | -5.77% | +28.96% | +5.50% | -32.23% | -32.87% |
| `broad_top5_vol20` | -2.71% | +1.84% | -0.92% | +54.54% | +9.60% | -21.42% | -22.01% |

The frozen `broad_top5_vol20` ends at 1545.37 USDT versus capped control's
2102.65 USDT. Its later-continuous net is -0.92% versus +35.45%.
A small development advantage does not establish a later profit advantage. No later
leader is substituted for the frozen choice; all rejected policies stay visible.

## Every-policy cost and timing stress

| Policy | Doubled-cost later net | Doubled-cost full net | Doubled-cost full DD | Delayed later net | Delayed full net | Delayed full DD |
|---|---:|---:|---:|---:|---:|---:|
| `spot_control` | +27.82% | +88.24% | -19.24% | +25.24% | +94.29% | -20.20% |
| `capped_control` | +31.23% | +93.80% | -19.32% | +28.12% | +99.54% | -20.37% |
| `broad_inverse_vol` | -4.97% | +34.49% | -28.19% | +1.36% | +66.56% | -26.95% |
| `broad_top3` | -17.22% | +42.79% | -38.37% | -15.44% | +57.05% | -37.58% |
| `broad_top3_vol` | -7.52% | +72.19% | -33.09% | -5.80% | +101.00% | -31.85% |
| `broad_top5_vol` | -4.52% | +34.46% | -28.81% | -4.76% | +40.80% | -29.07% |
| `broad_top5_consensus` | -9.12% | +16.97% | -34.89% | -7.26% | +27.01% | -33.19% |
| `broad_top5_vol20` | -3.87% | +43.33% | -23.13% | -1.43% | +58.53% | -22.99% |

The high/low bound assumes unfavorable within-bar/cross-asset ordering; it is not
exact tick chronology. The 20% historical selection budget is not a future maximum-loss guarantee.

## Costs, turnover and fill frequency

A fill is one buy/sell leg, not a completed round trip. Full nominal accounts.

| Policy | Fills | Active days | Fills/month | Turnover / initial | Fees USDT | Impact USDT | Net P&L USDT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | 432 | 271 | 7.58 | 73.85 | 73.85 | 44.31 | +1039.10 |
| `capped_control` | 448 | 287 | 7.86 | 76.94 | 76.94 | 46.17 | +1102.65 |
| `broad_inverse_vol` | 1140 | 432 | 20.00 | 102.31 | 102.31 | 61.38 | +532.01 |
| `broad_top3` | 653 | 328 | 11.46 | 98.31 | 98.31 | 58.99 | +586.53 |
| `broad_top3_vol` | 643 | 329 | 11.28 | 104.20 | 104.20 | 62.52 | +906.26 |
| `broad_top5_vol` | 701 | 322 | 12.30 | 69.39 | 69.39 | 41.63 | +459.73 |
| `broad_top5_consensus` | 771 | 355 | 13.53 | 70.22 | 70.22 | 42.13 | +289.61 |
| `broad_top5_vol20` | 764 | 320 | 13.40 | 62.59 | 62.59 | 37.55 | +545.37 |

All 104 paths independently reconcile signed cash flows, owned inventory
and NAV at every 4h close; largest residual 3.18e-12 USDT. Sixteen comparisons reproduce
both preceding nominal/cost controls' net, CAGR, DD, fees/impact and fill counts within 1e-8.
Cash/inventory stay nonnegative. Cost addback is diagnostic, not a no-cost account replay.

## Source, identity and missing-market audit

853 monthly archives plus 15 pre-period proofs are SHA-256 checked. 155,325 genuine bars and 735 absent asset/bars on the common grid.
Two documented terminal partial candles retain their actual halt closes. The AVAX duplicate
is identical in every field and corroborated by the entire day in a separate checksum-verified
daily archive (1 corroboration artifact). Both artifacts repeat the same row; this
is not an independent underlying market measurement. Raw hashes and row counts remain in JSON.

| Underlying research ID | Genuine bars | Absent bars | Absent ranges |
|---|---:|---:|---|
| `BTCUSDT` | 10404 | 0 | none |
| `ETHUSDT` | 10404 | 0 | none |
| `BNBUSDT` | 10404 | 0 | none |
| `SOLUSDT` | 10404 | 0 | none |
| `ADAUSDT` | 10404 | 0 | none |
| `XRPUSDT` | 10404 | 0 | none |
| `TERRA_CLASSIC` | 9689 | 715 | 2022-05-13 04:00:00+00:00 through 2022-09-09 04:00:00+00:00 (715) |
| `DOTUSDT` | 10404 | 0 | none |
| `AVAXUSDT` | 10404 | 0 | none |
| `DOGEUSDT` | 10404 | 0 | none |
| `SHIBUSDT` | 10404 | 0 | none |
| `POLYGON` | 10384 | 20 | 2024-09-10 04:00:00+00:00 through 2024-09-13 08:00:00+00:00 (20) |
| `UNIUSDT` | 10404 | 0 | none |
| `LTCUSDT` | 10404 | 0 | none |
| `LINKUSDT` | 10404 | 0 | none |

Source/indicator gaps stay NaN; only a flagged replay view uses zero sentinel marks.
No missing market can execute a fill. Owned quantity/cash persist; a held missing market is
conservatively valued at zero until actual quotes resume. This pessimistic assumption is not
a real observed price. 0/104 paths hold any unquoted inventory; any such
frozen-policy path would fail quality, irrespective of final profit. The known Polygon notice
takes effect only after the completed announcement day; missing days reset 90-day readiness.

## Short-horizon account gains

Full nominal continuous account with initial capital included. Overlapping calendar windows
are historical observations, not independent probabilities or future return promises.

| Policy | 7d doublings/windows | Best 7d | 30d doublings/windows | Best 30d | 90d doublings/windows | Best 90d |
|---|---:|---:|---:|---:|---:|---:|
| `spot_control` | 0/1728 | +10.39% | 0/1705 | +27.02% | 0/1645 | +29.50% |
| `capped_control` | 0/1728 | +11.58% | 0/1705 | +27.19% | 0/1645 | +29.74% |
| `broad_inverse_vol` | 0/1728 | +13.96% | 0/1705 | +35.22% | 0/1645 | +37.49% |
| `broad_top3` | 0/1728 | +28.09% | 0/1705 | +71.66% | 0/1645 | +80.68% |
| `broad_top3_vol` | 0/1728 | +27.80% | 0/1705 | +76.15% | 0/1645 | +80.26% |
| `broad_top5_vol` | 0/1728 | +19.58% | 0/1705 | +49.53% | 0/1645 | +51.34% |
| `broad_top5_consensus` | 0/1728 | +19.77% | 0/1705 | +48.16% | 0/1645 | +49.57% |
| `broad_top5_vol20` | 0/1728 | +17.27% | 0/1705 | +35.21% | 0/1645 | +41.62% |

## Descriptive uncertainty and frozen gates

Frozen versus capped control, continuous later nominal daily log-return differences:
annualized mean log excess -17.89%; 99% circular 14-day-block
percentile interval [-33.83%, -2.06%], 2000 replicates, seed 23812.
A positive lower bound is an additional historical gate. This interval is descriptive:
periods were already inspected, multiple rounds and universe selection are not accounted
for, and it is not confirmatory statistical significance. Unseen/paper evidence is required.

| Gate | Passed |
|---|:---:|
| development_eligible | true |
| new_candidate | true |
| accounting | true |
| controls_unchanged | true |
| beats_both_controls_2025 | false |
| beats_both_controls_2026 | false |
| beats_both_controls_later_continuous | false |
| beats_both_controls_full_continuous | false |
| no_held_unquoted_inventory | true |
| full_nominal_drawdown | false |
| full_doubled_drawdown | false |
| full_delayed_drawdown | false |
| later_doubled_positive | false |
| later_delayed_positive | false |
| descriptive_99pct_lower_positive | false |
| Material: full_cagr_1_5x | false |
| Material: later_profit_1_5x | false |

The material threshold additionally requires full CAGR at least +25.42% and later net at least +53.17%.
A material improvement is NOT supported by this experiment.

## Reproduction and evidence

```bash
python -m scripts.download_spot_universe
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_broad_universe
python -m scripts.write_broad_universe_report
python -m pytest -q tests
```

Checked raw source/cache/manifest: `data/raw/spot_universe/`. Frozen selection, targets, quote
mask and every equity/fill CSV: `artifacts/broad_spot_20261009/` (Git ignored). The original
three-asset source cache is reused without changing its checksums.
[THETABOT_BROAD_UNIVERSE_2026-10-09.json](THETABOT_BROAD_UNIVERSE_2026-10-09.json) retains every account summary, source URL/hash, alias rule, audit,
signal hash and gate; evidence SHA-256 `b83b626474336ceb6c183933df33e0c30a744fc4bc09216ad35b7741c228dbee`.

- Previously studied periods; not a new unseen holdout.
- Fixed dated 15-asset cohort reduces current-survivor bias but omits post-2021 entrants and other venues.
- Zero sentinel valuation for missing held markets is pessimistic, not an observed price; such paths fail quality gate.
- Old Terra 2.0 airdrops excluded; Polygon/old Terra quantity continuity uses documented identity rules.
- Model/universe selection and repeated-research uncertainty are not captured by the descriptive bootstrap.
- Modeled fees/impact, not actual fills, account-specific fee tiers or intrabar liquidity guarantees.
- Overlapping rolling windows do not estimate independent future doubling probabilities.
- Conservative within-bar high/low bound, not exact tick chronology.
- 20% historical budget does not guarantee future maximum loss.

Historical cohort: https://coinmarketcap.com/historical/20211226/
Primary identity references are linked in the protocol and stored in evidence; archive schema:
https://github.com/binance/binance-public-data .
