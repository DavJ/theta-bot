# Theta Bot: momentum refinement — 2026-10-09

**No improvement passes the fixed development-frozen screen.**
Frozen choice: `btc_regime`. Live eligibility: **false**.
The previously studied dates are reused; this is exploratory evidence, not a new unseen holdout.
No live order, default strategy, deployment or PR merge is enabled.

## Fixed design and execution

Six fixed changes to the existing 30-day momentum control: wider ordinary trading band,
three-day entry confirmation, stronger entry threshold, strength plus wider band, BTC regime
filter, and capped weight redistribution. Read [the frozen protocol](MOMENTUM_REFINEMENT_RESEARCH_PLAN.md)
and [the binding spot-only policy](SPOT_ONLY_POLICY.md). The implementation depends on PR #121.
Protocol SHA-256: `46eb49f28092b667368202a946217d14e96dc641df44055a37d76419dabc7a23`; written before new economic results.

The same 171 checked official spot archives supply 31,212 complete 4h bars for BTC, ETH and BNB
from January 2022 through September 2026. No external predictor or missing-bar interpolation
is added. Completed daily signals are published with the day's last 4h close and used at the
next daily open. Ordinary rebalancing is Monday; inactive exits and cap sales bypass the
ordinary band. The wider band is 3 percentage points, versus the control's 1 percentage point.

All targets retain 50% aggregate/25% per-asset caps in one shared 1,000-USDT account. Holdings
can drift between daily decisions. Sells precede buys, cash/inventory remain nonnegative
including fees, and no loss-doubling, leverage, borrowing, short or derivative is used.
Redistribution can deploy more of the same permitted budget; it does not raise the caps.
The entry state is signal eligibility, not a record of filled holdings. The strength threshold
is a fixed past-return heuristic, not a predicted future net gain or significance test.

Nominal costs per fill: 0.1% fee +5bps slippage +half a 2bps spread. Stress doubles the fee and
slippage, with spread unchanged. Impact is already in the fill price and is not charged twice.
Period-end inventory remains marked to market. Peaks and losses persist on continuous paths.

## Development choice

Choose the highest positive continuous 2022–2024 net profit with conservative 4h drawdown
no worse than -20%, including the control. Persist the choice before later economic replay.

| Variant | Development net | Development DD bound | Eligible |
|---|---:|---:|:---:|
| `spot_control` | +54.33% | -17.86% | yes |
| `wide_band` | +54.41% | -17.84% | yes |
| `confirmed_entry` | +19.13% | -23.66% | no |
| `strong_entry` | +22.33% | -22.97% | no |
| `strong_entry_wide` | +22.60% | -23.29% | no |
| `btc_regime` | +62.72% | -17.72% | yes |
| `capped_allocation` | +54.81% | -18.00% | yes |

## Net account results

2025 and Jan–Sep 2026 start independent accounts. The two continuous columns carry all
losses, fills and compounding forward. Later results cannot select a replacement winner.

| Variant | 2025 net | Jan–Sep 2026 net | Continuous 2025–Sep 2026 | Full 2022–Sep 2026 | Full CAGR | Full DD bound | Full stress net |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | +19.59% | +10.18% | +31.75% | +103.91% | +16.19% | -18.54% | +88.24% |
| `wide_band` | +19.60% | +9.97% | +31.56% | +103.66% | +16.16% | -18.59% | +88.29% |
| `confirmed_entry` | +15.63% | +3.11% | +19.04% | +41.96% | +7.66% | -23.66% | +33.96% |
| `strong_entry` | +19.90% | +1.63% | +21.65% | +49.09% | +8.78% | -22.97% | +40.07% |
| `strong_entry_wide` | +19.91% | +1.29% | +21.56% | +49.26% | +8.80% | -23.29% | +40.34% |
| `btc_regime` | +15.93% | -0.04% | +15.70% | +88.68% | +14.31% | -17.72% | +75.15% |
| `capped_allocation` | +21.04% | +12.02% | +35.45% | +110.26% | +16.95% | -18.62% | +93.80% |

Frozen `btc_regime` ends at 1886.78 USDT; control at 2039.10 USDT.
Its continuous later return differs from control by -16.05 percentage points.
Full 4h-close DD is -16.67%, conservative high/low bound -17.72%.

The high/low bound assumes unfavorable cross-asset/intracandle ordering; it is not exact
tick drawdown. The 20% historical threshold does not guarantee a maximum future loss.

The descriptive `capped_allocation` comparison improves all four nominal period returns
relative to control. It uses the same signal with more complete deployment within the same
caps, rather than an additional predictive source. It was not the development-frozen choice
and cannot replace the failed BTC-regime selection after inspecting later data. Its higher
historical profit is an exploratory finding that needs subsequent unseen/paper confirmation.
The prescribed timing stress covers the frozen choice and control, not this later leader.

## Costs and trading frequency

Full nominal continuous account. A fill is one executed buy/sell leg, not a round trip.

| Variant | Fills | Active trading days | Fills/month | Turnover / initial capital | Fees USDT | Impact USDT | Net P&L USDT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spot_control` | 432 | 271 | 7.58 | 73.85 | 73.85 | 44.31 | +1039.10 |
| `wide_band` | 399 | 256 | 7.00 | 72.78 | 72.78 | 43.67 | +1036.55 |
| `confirmed_entry` | 322 | 208 | 5.65 | 42.05 | 42.05 | 25.23 | +419.64 |
| `strong_entry` | 330 | 217 | 5.79 | 46.64 | 46.64 | 27.99 | +490.88 |
| `strong_entry_wide` | 315 | 210 | 5.53 | 46.27 | 46.27 | 27.76 | +492.58 |
| `btc_regime` | 400 | 204 | 7.02 | 72.50 | 72.50 | 43.50 | +886.78 |
| `capped_allocation` | 448 | 287 | 7.86 | 76.94 | 76.94 | 46.17 | +1102.65 |

All 71 paths independently reconcile cash, owned inventory and NAV at every 4h close.
Largest NAV residual: 2.96e-12 USDT. The eight control checks reproduce the preceding
report's profits/costs/DD and the original daily engine's equity and fill quantities/prices/fees/impact
within 1e-8 absolute tolerance. No borrowing or financial short is hidden by inventory clamping.
Cost addback is diagnostic; it is not the result of a hypothetical no-cost replay.

## Short-horizon gains

Continuous full nominal account, rolling calendar-day closing endpoints including initial NAV.
Windows overlap; counts are historical observations, not independent future probabilities.

| Variant | 7d doublings / windows | Best 7d | 30d doublings / windows | Best 30d | 90d doublings / windows | Best 90d |
|---|---:|---:|---:|---:|---:|---:|
| `spot_control` | 0 / 1728 | +10.39% | 0 / 1705 | +27.02% | 0 / 1645 | +29.50% |
| `wide_band` | 0 / 1728 | +10.39% | 0 / 1705 | +27.02% | 0 / 1645 | +29.50% |
| `confirmed_entry` | 0 / 1728 | +10.00% | 0 / 1705 | +27.40% | 0 / 1645 | +29.92% |
| `strong_entry` | 0 / 1728 | +10.00% | 0 / 1705 | +27.40% | 0 / 1645 | +29.92% |
| `strong_entry_wide` | 0 / 1728 | +10.00% | 0 / 1705 | +27.40% | 0 / 1645 | +29.92% |
| `btc_regime` | 0 / 1728 | +9.96% | 0 / 1705 | +27.17% | 0 / 1645 | +31.37% |
| `capped_allocation` | 0 / 1728 | +11.58% | 0 / 1705 | +27.19% | 0 / 1645 | +29.74% |

## One-bar timing stress and improvement gates

An extra 4h signal delay moves both target availability and decisions to 04:00 UTC.
The frozen choice and control are delayed identically; this is a hypothetical timing stress,
not a measured network delay. Timing results do not choose another winner.

| Variant | Period | Delayed net | Delayed DD bound |
|---|---|---:|---:|
| `spot_control` | 2025 | +15.83% | -13.50% |
| `spot_control` | 2026 | +8.19% | -10.99% |
| `spot_control` | later_continuous | +25.24% | -20.27% |
| `spot_control` | full_continuous | +94.29% | -20.20% |
| `btc_regime` | 2025 | +12.34% | -13.50% |
| `btc_regime` | 2026 | -0.66% | -9.01% |
| `btc_regime` | later_continuous | +11.60% | -17.33% |
| `btc_regime` | full_continuous | +77.28% | -17.28% |

| Development-frozen improvement gate | Passed |
|---|:---:|
| development_eligible | true |
| new_candidate | true |
| accounting | true |
| beats_control_2025 | false |
| beats_control_2026 | false |
| beats_control_later_continuous | false |
| beats_control_full_continuous | false |
| full_nominal_drawdown | true |
| full_doubled_drawdown | true |
| later_doubled_positive | true |

Historical improvement screen: **False**. No improvement passes the fixed development-frozen screen.
A control selected again cannot count as an improvement. Any later-period leader is descriptive
unless it was also development-frozen and passes every fixed gate. No live eligibility is inferred.

## Reproduction and limitations

```bash
python -m scripts.download_spot_flow
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_momentum_refinement
python -m scripts.write_momentum_refinement_report
python -m pytest -q tests
```

Source archives, manifest, frozen selection and every equity/trade CSV are reproducible under
`data/raw/spot_flow/` and `artifacts/momentum_refinement_20261009/` (ignored by Git). This committed
[THETABOT_MOMENTUM_REFINEMENT_2026-10-09.json](THETABOT_MOMENTUM_REFINEMENT_2026-10-09.json) retains all summaries, source URLs/hashes, gates and control checks.
Evidence JSON SHA-256: `1484c8d9df93ff470840570920405e14613275bd0b7dfa9b9e263ac3f3ccbaf6`.

- Previously studied periods; not a fresh holdout.
- Three surviving assets; universe survivorship not controlled.
- Fixed entry-strength heuristic, not a forecast of future net returns or a significance test.
- Modeled fees and impact, not measured live fills or account-specific tiers.
- Signal eligibility state is distinct from actual filled holdings.
- Overlapping rolling windows do not estimate independent future doubling probabilities.
- Conservative within-bar high/low bound, not exact tick drawdown.
- 20% historical budget does not guarantee future maximum loss.

Primary documentation: [Binance public spot archive schema/checksums](https://github.com/binance/binance-public-data).
