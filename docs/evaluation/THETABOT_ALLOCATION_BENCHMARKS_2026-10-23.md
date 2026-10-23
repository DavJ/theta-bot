# Guarded momentum versus fixed allocations — 2026-10-23

**Known history; fixed existing candidate, no fresh holdout or signal search.**

Historical incremental-value screen: **True**.

Same execution and requested volatility guard; realized risk differs.
Equal/inverse-volatility references rebalance weekly and sell for risk; they are not buy-and-hold.
Full: January 2022–September 2026. Later: independently funded January 2025–September 2026.
Net includes modeled fills, fees, spread and slippage, before tax and operating costs.

## Initial 1000 USDT

| Policy | Full nominal net | Full joint net | Worst full DD | Nominal realized vol | Mean exposure | Fees + impact |
|---|---:|---:|---:|---:|---:|---:|
| guarded_edge | +84.52% | +71.28% | -15.24% | 11.82% | 17.51% | 84.28 |
| cash | +0.00% | +0.00% | 0.00% | 0.00% | 0.00% | 0.00 |
| equal_25 | +32.58% | +33.73% | -19.92% | 12.18% | 23.33% | 1.57 |
| equal_50 | +50.36% | +50.66% | -28.91% | 16.56% | 31.81% | 9.81 |
| inverse_vol_50 | +53.13% | +50.14% | -29.18% | 16.81% | 33.26% | 13.32 |

| Policy | Later nominal net | Later joint net | Worst later DD | Later nominal realized vol |
|---|---:|---:|---:|---:|
| guarded_edge | +31.16% | +23.86% | -15.29% | 11.89% |
| cash | +0.00% | +0.00% | 0.00% | 0.00% |
| equal_25 | +3.98% | +3.72% | -18.26% | 11.38% |
| equal_50 | +1.59% | +2.50% | -28.62% | 17.03% |
| inverse_vol_50 | +3.81% | +3.89% | -28.77% | 17.29% |

Realized vol is annualized population daily NAV-return standard deviation, including the initial funded return.

## Initial 10000 USDT

| Policy | Full nominal net | Full joint net | Worst full DD | Nominal realized vol | Mean exposure | Fees + impact |
|---|---:|---:|---:|---:|---:|---:|
| guarded_edge | +83.37% | +70.47% | -15.03% | 11.65% | 17.32% | 857.55 |
| cash | +0.00% | +0.00% | 0.00% | 0.00% | 0.00% | 0.00 |
| equal_25 | +31.84% | +32.75% | -19.83% | 12.28% | 23.49% | 24.29 |
| equal_50 | +50.39% | +49.75% | -28.30% | 16.40% | 31.54% | 161.19 |
| inverse_vol_50 | +50.90% | +50.09% | -28.68% | 16.63% | 32.89% | 184.90 |

| Policy | Later nominal net | Later joint net | Worst later DD | Later nominal realized vol |
|---|---:|---:|---:|---:|
| guarded_edge | +30.78% | +23.48% | -15.04% | 11.73% |
| cash | +0.00% | +0.00% | 0.00% | 0.00% |
| equal_25 | +3.54% | +3.45% | -18.66% | 11.79% |
| equal_50 | +2.52% | +1.75% | -28.26% | 16.80% |
| inverse_vol_50 | +3.12% | +2.42% | -28.71% | 17.06% |

Realized vol is annualized population daily NAV-return standard deviation, including the initial funded return.

## Descriptive incremental operating-cost headroom

Candidate terminal NAV minus each reference NAV, divided by period calendar months.
10,000-USDT accounts; USDT per month. Negative means the candidate already trails.
Only additional candidate expenses relative to the reference belong here. No recurring withdrawals are replayed.

| Reference | Full nominal | Full joint | Later nominal | Later joint |
|---|---:|---:|---:|---:|
| equal_25 | +90.46 | +66.20 | +129.95 | +95.54 |
| equal_50 | +57.89 | +36.38 | +134.85 | +103.64 |
| inverse_vol_50 | +56.99 | +35.78 | +131.96 | +100.46 |

## Evidence and fixed gates

- 171 official monthly archives checked; 24 archived candidate paths reproduce.
- 120 full funded paths reconcile nonnegative cash, inventory and NAV.
- 10 CSV accounts independently audited: 104,040 bars and 13,156 fills.
- Maximum independent NAV error: 1.34e-09 USDT.
- Protocol SHA-256: `2b3ed53e1f24d375d16efbc30dc16dc897a65e37fcc3054c2647418ed36376cd`.

- accounting: **True**
- candidate_all_drawdowns_within_budget: **True**
- candidate_later_full_positive_all_stresses: **True**
- candidate_beats_each_invested_reference_later_full_nominal_joint: **True**

## Limits

- Previously studied dates and surviving three-asset universe; no fresh holdout.
- Shared guard settings do not exactly match realized volatility, exposure or drawdown.
- References rebalance weekly with risk sales; they are not literal buy-and-hold.
- Terminal surplus per month is descriptive incremental cost headroom, not cash withdrawals or forecast income.
- No order-book, executable paper fills, tax, operating-cost, custody or quote-FX model.
- USDT archives do not establish executable EEA/account markets.
- Terminal inventory is marked without forced liquidation; historical risk settings do not cap future loss.

Live eligibility remains false. [Fixed plan](ALLOCATION_BENCHMARK_RESEARCH_PLAN.md).

## Reproduce

```bash
python -m scripts.download_spot_flow
python -m scripts.research_allocation_benchmarks freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_allocation_benchmarks run --jobs 4
```
