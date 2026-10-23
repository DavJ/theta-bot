# ThetaBot: turnover and actual-holdings risk — 2026-10-13

**The policy `guarded_edge` was fixed before development payoffs.**
All dates were already studied. These are modeled net returns, not promised income.

Ordinary trades can stop at the nearest 1pp band edge. Optional risk checks reduce
actual inventory at every 4h open using the prior completed-day 60-day covariance
with 25% diagonal shrinkage and a fixed 15% annualized volatility ceiling.
Target caps remain 50% total/25% per asset; the guard never adds leverage or cash.

Protocol SHA-256: `0976c1e625e3e93c0a464e0117ad5f80b13ce8c6dccbfa6efcf6f26f649d0cc4`. [Fixed plan](COST_RISK_RESEARCH_PLAN.md).

## Initial account 1000 USDT

Profit-improvement screen: **False**; risk-improvement screen: **True**.

| Policy | Nominal full net | Full joint stress net | Worst full DD | Full CAGR | Full fills | Fees + impact |
|---|---:|---:|---:|---:|---:|---:|
| `spot_control` | +110.26% | +84.16% | -21.30% | +16.95% | 448 | 123.11 |
| `band_edge` | +107.10% | +82.03% | -20.71% | +16.57% | 387 | 114.92 |
| `vol_guard` | +85.28% | +71.44% | -15.32% | +13.87% | 492 | 85.99 |
| `guarded_edge` | +84.52% | +71.28% | -15.24% | +13.77% | 499 | 84.28 |

| Policy | 2025 nominal | 2026 nominal | Later continuous nominal |
|---|---:|---:|---:|
| `spot_control` | +21.04% | +12.02% | +35.45% |
| `band_edge` | +21.31% | +11.52% | +35.28% |
| `vol_guard` | +18.50% | +10.70% | +31.17% |
| `guarded_edge` | +18.72% | +10.60% | +31.16% |

Independent yearly accounts reset funding; continuous columns carry all holdings/losses.
Percentages across separately funded years must not be added.

| Fixed-candidate gate | Passed |
|---|:---:|
| accounting | True |
| all_historical_drawdowns_within_budget | True |
| later_and_full_stresses_profitable | True |
| lower_worst_full_drawdown | True |
| beats_nominal_control_2025 | False |
| beats_nominal_control_2026 | False |
| beats_nominal_control_later_continuous | False |
| beats_nominal_control_full_continuous | False |
| beats_full_joint_control | False |

| Full scenario | Guarded-edge net | DD bound | Risk reduction fills | Residual bars |
|---|---:|---:|---:|---:|
| nominal | +84.52% | -13.29% | 176 | 2003 |
| cost_stress | +73.17% | -14.27% | 171 | 2073 |
| delayed | +81.67% | -14.17% | 178 | 2072 |
| joint_stress | +71.28% | -15.24% | 173 | 2134 |

Residual bars disclose the actual holdings left above the requested constrained target,
including the 10-USDT minimum and costs. A requested cap is not an enforced future risk guarantee.

## Initial account 10000 USDT

Profit-improvement screen: **False**; risk-improvement screen: **True**.

| Policy | Nominal full net | Full joint stress net | Worst full DD | Full CAGR | Full fills | Fees + impact |
|---|---:|---:|---:|---:|---:|---:|
| `spot_control` | +108.82% | +84.20% | -21.16% | +16.78% | 818 | 1238.57 |
| `band_edge` | +105.76% | +81.79% | -20.63% | +16.41% | 638 | 1152.51 |
| `vol_guard` | +83.42% | +70.56% | -15.09% | +13.63% | 1180 | 866.46 |
| `guarded_edge` | +83.37% | +70.47% | -15.03% | +13.62% | 1321 | 857.55 |

| Policy | 2025 nominal | 2026 nominal | Later continuous nominal |
|---|---:|---:|---:|
| `spot_control` | +20.91% | +11.95% | +35.31% |
| `band_edge` | +21.12% | +11.59% | +35.16% |
| `vol_guard` | +18.41% | +10.56% | +30.95% |
| `guarded_edge` | +18.60% | +10.28% | +30.78% |

Independent yearly accounts reset funding; continuous columns carry all holdings/losses.
Percentages across separately funded years must not be added.

| Fixed-candidate gate | Passed |
|---|:---:|
| accounting | True |
| all_historical_drawdowns_within_budget | True |
| later_and_full_stresses_profitable | True |
| lower_worst_full_drawdown | True |
| beats_nominal_control_2025 | False |
| beats_nominal_control_2026 | False |
| beats_nominal_control_later_continuous | False |
| beats_nominal_control_full_continuous | False |
| beats_full_joint_control | False |

| Full scenario | Guarded-edge net | DD bound | Risk reduction fills | Residual bars |
|---|---:|---:|---:|---:|
| nominal | +83.37% | -13.06% | 916 | 944 |
| cost_stress | +72.28% | -14.06% | 907 | 957 |
| delayed | +81.37% | -14.05% | 921 | 966 |
| joint_stress | +70.47% | -15.03% | 911 | 986 |

Residual bars disclose the actual holdings left above the requested constrained target,
including the 10-USDT minimum and costs. A requested cap is not an enforced future risk guarantee.

## Independent evidence and limits

All 171 official archive bytes/checksums were rechecked without network access.
All 160 paths reconcile cash, nonnegative owned inventory and NAV.
All 40 archived control paths reproduce; maximum NAV error 5.09e-11 USDT.
The evidence ZIP contains 32 complete continuous accounts (64 CSV files), source manifest,
protocol/selection timestamps, runtime versions and SHA-256 hashes for every CSV.
Ablations diagnose the intervention and do not replace the fixed candidate after results.

A separate CSV audit checks 32 accounts, 332,928 bars and 23,041 fills.
It recomputes source-open fill costs, NAV and conservative DD; direct NumPy return-window covariance verifies guard mathematics.
Maximum independent NAV residual: 7.13e-10 USDT; volatility residual: 5.48e-14.

- Known dates and surviving three-asset universe, not independent alpha evidence.
- Risk uses lagged completed daily covariance and can only sell on modeled open fills.
- Signal/covariance lag stress is not a model of delayed emergency-order execution.
- Minimum notionals and fill costs can leave actual volatility above its requested target.
- 20% historical drawdown and 15% estimated volatility are not future loss guarantees.
- No tick path, historical order-book depth, taxes, custody/default risk or USDT FX risk.
- Terminal inventory is marked, not forcibly liquidated; costs use fixed hypothetical rates.

Live eligibility is false. No exchange connection, live/default selection, deployment or merge is enabled.

Repeated selection on known data can overfit: [Bailey et al., original paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

## Reproduce

```bash
python -m scripts.download_spot_flow
python -m scripts.research_cost_risk --freeze-only
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_cost_risk --jobs 4 --report-out docs/evaluation
```

## Download the complete audit archive

The ZIP is stored in three byte-preserving parts to fit connector request limits.
The [manifest](THETABOT_COST_RISK_2026-10-13_EVIDENCE.zip.manifest.json) records each part and the complete archive SHA-256.
Concatenate parts in order, verify that SHA-256, then extract.

```bash
cat docs/evaluation/THETABOT_COST_RISK_2026-10-13_EVIDENCE.zip.part001 docs/evaluation/THETABOT_COST_RISK_2026-10-13_EVIDENCE.zip.part002 docs/evaluation/THETABOT_COST_RISK_2026-10-13_EVIDENCE.zip.part003 > /tmp/thetabot-cost-risk-evidence.zip
python -m zipfile -t /tmp/thetabot-cost-risk-evidence.zip
```
