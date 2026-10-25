# Fixed momentum-entry hysteresis — 2026-10-25

**Previously studied history; fixed candidate, no new holdout or parameter search.**

Historical profit screen: **True**.
Cost/turnover hypothesis supported: **True**.

Candidate enters above 0.64% completed 30-day momentum and stays eligible until momentum is nonpositive.
All variants keep the same 15% requested volatility guard, caps, 1pp band-edge execution and modeled costs.
Full: January 2022–September 2026; later: independently funded January 2025–September 2026.
Net is after modeled trading costs, before tax and operating costs.

## Initial 1000 USDT

| Policy | Full nominal | Full joint stress | Worst full DD | Later nominal | Later joint stress |
|---|---:|---:|---:|---:|---:|
| guarded_edge | +84.52% | +71.28% | -15.24% | +31.16% | +23.86% |
| entry_hysteresis | +86.07% | +72.26% | -15.25% | +31.44% | +24.06% |
| symmetric_hurdle | +82.08% | +68.49% | -14.56% | +31.66% | +23.44% |
| two_day_confirmation | +60.64% | +51.70% | -18.93% | +29.74% | +23.64% |
| hysteresis_confirmed | +61.38% | +52.21% | -18.46% | +29.39% | +23.67% |

| Policy | Full fees + impact | Full fills | Risk-reduction fills | Mean exposure | Realized annual vol |
|---|---:|---:|---:|---:|---:|
| guarded_edge | 84.28 | 499 | 176 | 17.51% | 11.82% |
| entry_hysteresis | 81.62 | 487 | 180 | 17.38% | 11.80% |
| symmetric_hurdle | 85.57 | 483 | 174 | 16.73% | 11.67% |
| two_day_confirmation | 64.11 | 428 | 148 | 16.68% | 11.53% |
| hysteresis_confirmed | 63.26 | 423 | 147 | 16.57% | 11.52% |

## Initial 10000 USDT

| Policy | Full nominal | Full joint stress | Worst full DD | Later nominal | Later joint stress |
|---|---:|---:|---:|---:|---:|
| guarded_edge | +83.37% | +70.47% | -15.03% | +30.78% | +23.48% |
| entry_hysteresis | +85.19% | +71.42% | -15.03% | +31.00% | +23.68% |
| symmetric_hurdle | +81.39% | +67.52% | -14.31% | +31.12% | +23.08% |
| two_day_confirmation | +60.12% | +51.72% | -18.70% | +29.50% | +23.37% |
| hysteresis_confirmed | +60.85% | +52.07% | -18.23% | +29.15% | +23.40% |

| Policy | Full fees + impact | Full fills | Risk-reduction fills | Mean exposure | Realized annual vol |
|---|---:|---:|---:|---:|---:|
| guarded_edge | 857.55 | 1321 | 916 | 17.32% | 11.65% |
| entry_hysteresis | 832.25 | 1299 | 913 | 17.20% | 11.63% |
| symmetric_hurdle | 872.01 | 1283 | 895 | 16.54% | 11.50% |
| two_day_confirmation | 655.44 | 1142 | 787 | 16.48% | 11.37% |
| hysteresis_confirmed | 647.09 | 1145 | 793 | 16.37% | 11.36% |

## Fixed gates and evidence

- accounting: **True**
- candidate_all_drawdowns_within_budget: **True**
- candidate_later_full_positive_all_stresses: **True**
- candidate_beats_control_later_full_nominal_joint: **True**
- lower_candidate_costs: **True**
- fewer_candidate_fills: **True**
- 171 official monthly archives rechecked; 24 archived control paths reproduce.
- 120 funded paths reconcile cash, owned inventory and NAV.
- 10 complete CSV accounts independently audited: 104,040 bars and 12,398 fills.
- Maximum independent NAV residual: 7.68e-10 USDT.
- Protocol SHA-256: `ada20d1d9955964d272f820e5ecc9269edca962d9a61ba6fcb7ce1d7100ee032`.

## Limits

- Known dates and surviving three-asset universe; no fresh holdout.
- Trailing 30-day strength is not an estimated future net edge; eligibility state is not filled inventory.
- Diagnostics do not replace the fixed candidate after observing outcomes.
- Shared requested risk settings do not ensure equal realized exposure or volatility.
- Other-cost annotations include all non-risk fills, including required cap sales.
- Modeled opens, fixed fees and terminal marks; no order-book, account/EEA, tax, operating-cost, custody or quote-FX validation.
- The historical 20% drawdown screen and 15% volatility request do not cap future losses.

Live eligibility remains false. [Frozen plan](ENTRY_HYSTERESIS_RESEARCH_PLAN.md).

## Reproduction receipt

The source and protocol were published in `69d0116224ade4f7e2e0e3af90deb4525d89f069`
before the first payoffs. Transient outputs were subsequently lost and regenerated
from that exact source/runtime/protocol on October 25. All 171 archive and three
CSV identities reproduce. The evidence includes `reproduction_receipt.json`;
the development-selection timestamp is newly generated, not backdated.

## Reproduce

```bash
python -m scripts.download_spot_flow
python -m scripts.research_entry_hysteresis freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_entry_hysteresis run --jobs 4
```
