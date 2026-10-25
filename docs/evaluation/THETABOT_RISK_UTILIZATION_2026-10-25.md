# Fixed risk utilization — 2026-10-25

**Adaptive follow-up on reused history; greater risk utilization is not new alpha.**

Fixed candidate `entry_20` historical screen: **True**.

Full: January 2022–September 2026; later: independently funded January 2025–September 2026.
Net is after modeled trading costs, before tax and operating costs. Terminal holdings are marked.

## Initial 1000 USDT

| Policy | Full nominal | Full joint stress | Worst full DD | Nominal CAGR | Later nominal | Later joint |
|---|---:|---:|---:|---:|---:|---:|
| control_15 | +84.52% | +71.28% | -15.24% | 13.77% | +31.16% | +23.86% |
| entry_15 | +86.07% | +72.26% | -15.25% | 13.97% | +31.44% | +24.06% |
| entry_175 | +97.14% | +79.70% | -17.18% | 15.37% | +34.53% | +25.63% |
| entry_20 | +98.71% | +78.60% | -18.69% | 15.56% | +35.06% | +25.11% |

| Policy | Requested vol | Realized annual vol | Mean exposure | Fees + impact | Fills |
|---|---:|---:|---:|---:|---:|
| control_15 | 15.0% | 11.82% | 17.51% | 84.28 | 499 |
| entry_15 | 15.0% | 11.80% | 17.38% | 81.62 | 487 |
| entry_175 | 17.5% | 13.26% | 19.54% | 91.87 | 474 |
| entry_20 | 20.0% | 14.43% | 21.14% | 97.64 | 442 |

## Initial 10000 USDT

| Policy | Full nominal | Full joint stress | Worst full DD | Nominal CAGR | Later nominal | Later joint |
|---|---:|---:|---:|---:|---:|---:|
| control_15 | +83.37% | +70.47% | -15.03% | 13.62% | +30.78% | +23.48% |
| entry_15 | +85.19% | +71.42% | -15.03% | 13.86% | +31.00% | +23.68% |
| entry_175 | +96.10% | +79.07% | -17.09% | 15.24% | +34.45% | +26.10% |
| entry_20 | +98.13% | +78.28% | -18.56% | 15.49% | +35.10% | +25.47% |

| Policy | Requested vol | Realized annual vol | Mean exposure | Fees + impact | Fills |
|---|---:|---:|---:|---:|---:|
| control_15 | 15.0% | 11.65% | 17.32% | 857.55 | 1321 |
| entry_15 | 15.0% | 11.63% | 17.20% | 832.25 | 1299 |
| entry_175 | 17.5% | 13.12% | 19.36% | 932.14 | 1200 |
| entry_20 | 20.0% | 14.28% | 20.95% | 989.07 | 1026 |

## Fixed gates and evidence

The fixed 20% candidate passes against both 15% controls, but it is not uniformly
best. The 17.5% diagnostic earns more in combined stress with lower worst
drawdown at both capitals. It remains diagnostic; there is no post-result
replacement of the fixed candidate. At 10,000 USDT, increasing entry-15 to
entry-20 raises nominal realized volatility from 11.63% to 14.28%, mean exposure
from 17.20% to 20.95% and modeled fees/impact from 832.25 to 989.07 USDT. The
gain is greater risk utilization, not evidence of a new predictive signal.

- accounting: **True**
- candidate_all_drawdowns_within_budget: **True**
- candidate_later_full_positive_all_stresses: **True**
- candidate_beats_both_15_controls_later_full_nominal_joint: **True**
- 171 official archives rechecked; 48 preceding 15% controls reproduce.
- 96 funded paths reconcile; 8 complete raw CSV accounts independently audited.
- Independent audit covers 83,232 bars and 9,714 fills.
- Maximum independent NAV residual: 7.13e-10 USDT.
- Maximum independent volatility residual: 3.61e-16.
- Protocol SHA-256: `33bacc3a9b44b00e2e05825e49cb031693ceae50a9a53a2d3a33459bd27b9e78`.
- Preceding report SHA-256: `f9d5c5f4b77ba7e9c2eb96c79616d3698d9a5755c9741f8def82a6ece63ef90b`.

## Limits

- Adaptive follow-up on previously studied dates and three surviving assets; no fresh holdout.
- Greater fixed risk utilization changes exposure and loss risk, not predictive information.
- No diagnostic replaces the fixed candidate after results.
- Modeled opens, fixed fees and terminal marks; no executable/account/EEA, tax, operating-cost, custody or quote-FX validation.
- Historical 20% drawdown selection and annual-volatility requests do not cap future losses.

Live eligibility remains false. [Frozen plan](RISK_UTILIZATION_RESEARCH_PLAN.md).

## Reproduce

```bash
python -m scripts.download_spot_flow
python -m scripts.research_entry_hysteresis freeze
python -m scripts.research_entry_hysteresis run --jobs 4
python -m scripts.research_risk_utilization freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_risk_utilization run --jobs 4
```
