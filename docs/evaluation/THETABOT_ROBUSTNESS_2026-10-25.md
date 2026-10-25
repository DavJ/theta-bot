# Fixed calendar and asset-omission audit — 2026-10-25

**Adaptive diagnostic on reused history; no policy or universe selection.**

Fixed `entry_20` omission screen: **True**.

Each account starts with 10,000 USDT. Full: January 2022–September 2026;
later: independently funded January 2025–September 2026. Net includes modeled
trading costs, excludes operating costs/tax, and marks terminal inventory.

## Fixed candidate with one asset omitted

| Universe | Full nominal | Full joint | Worst full DD | Later nominal | Later joint |
|---|---:|---:|---:|---:|---:|
| all_three | +98.13% | +78.28% | -18.56% | +35.10% | +25.47% |
| without_BTCUSDT | +79.88% | +61.63% | -18.10% | +37.94% | +29.12% |
| without_ETHUSDT | +80.52% | +67.22% | -18.31% | +21.89% | +15.96% |
| without_BNBUSDT | +78.61% | +62.09% | -17.63% | +21.18% | +13.05% |

Eligible retained assets each receive up to the existing 25% cap before risk scaling;
this reallocates the remaining budget and is not subtraction of historical asset P&L.

## Rolling calendar years in the original universe

| Policy | Scenario | Worst 12 months | Negative 12-month windows | Worst month | Best-3 positive log-return share | Return excluding best 3 months* |
|---|---|---:|---:|---:|---:|---:|
| control_15 | nominal | -5.80% | 5/46 | -7.69% | 28.72% | +37.12% |
| control_15 | joint_stress | -4.29% | 6/46 | -5.83% | 30.14% | +27.76% |
| entry_15 | nominal | -4.65% | 4/46 | -7.69% | 28.85% | +38.48% |
| entry_15 | joint_stress | -3.76% | 5/46 | -5.83% | 30.25% | +28.48% |
| entry_175 | nominal | -5.35% | 5/46 | -8.97% | 28.94% | +41.72% |
| entry_175 | joint_stress | -4.31% | 6/46 | -6.82% | 30.37% | +29.72% |
| entry_20 | nominal | -5.98% | 6/46 | -10.11% | 29.23% | +40.04% |
| entry_20 | joint_stress | -4.90% | 7/46 | -7.81% | 30.58% | +26.48% |

*Ex-post mathematical concentration diagnostic, not a tradable monthly switch.
Windows overlap and are not independent observations or future loss probabilities.

## Calendar-year returns in the original funded accounts

Each row uses one continuous full account; 2026 covers January–September only.
The full accounts begin in cash and require 60 completed daily returns for
sizing. Entry-20 first purchases on **2022-03-21**; the 2022 result includes
this initial cash/warmup phase. A mature account with earlier price history
could behave differently. Independently funded later accounts have preceding
price history, but inherit no cash or positions from development.

| Policy | Scenario | 2022 | 2023 | 2024 | 2025 | 2026 Jan–Sep |
|---|---|---:|---:|---:|---:|---:|
| control_15 | nominal | -5.80% | +22.33% | +21.63% | +18.64% | +10.27% |
| control_15 | joint_stress | -4.29% | +18.22% | +22.01% | +14.31% | +8.01% |
| entry_15 | nominal | -4.65% | +21.85% | +21.62% | +18.69% | +10.42% |
| entry_15 | joint_stress | -3.76% | +18.27% | +21.77% | +14.50% | +8.01% |
| entry_175 | nominal | -5.35% | +23.45% | +24.74% | +20.85% | +11.33% |
| entry_175 | joint_stress | -4.31% | +19.49% | +24.16% | +16.17% | +8.58% |
| entry_20 | nominal | -5.98% | +24.00% | +25.70% | +20.78% | +11.94% |
| entry_20 | joint_stress | -4.90% | +19.87% | +24.60% | +15.32% | +8.84% |

## Nominal additive asset P&L attribution (USDT)

| Universe | Policy | BTC net contribution | ETH net contribution | BNB net contribution | Total net |
|---|---|---:|---:|---:|---:|
| all_three | control_15 | +2666.12 | +2902.09 | +2768.77 | +8336.98 |
| all_three | entry_15 | +2703.55 | +3014.32 | +2801.07 | +8518.94 |
| all_three | entry_175 | +3057.64 | +3262.93 | +3289.67 | +9610.24 |
| all_three | entry_20 | +3044.37 | +3308.10 | +3460.07 | +9812.54 |
| without_BTCUSDT | entry_20 | +0.00 | +4338.69 | +3649.34 | +7988.03 |
| without_ETHUSDT | entry_20 | +3964.55 | +0.00 | +4087.69 | +8052.24 |
| without_BNBUSDT | entry_20 | +3823.82 | +4037.56 | +0.00 | +7861.39 |

Attribution includes holding gains/losses, fees and adverse execution impact.
These are contributions to the shared-cash account, not stand-alone asset returns.

## Fixed gates and verification

- accounting: **True**
- original_control_reproduction: **True**
- candidate_omission_drawdowns_within_budget: **True**
- candidate_omission_later_full_positive: **True**
- 171 official archives rechecked; 24 preceding summaries reproduce.
- 96 funded paths reconcile; 14 complete CSV accounts independently audited.
- Audit covers 145,656 bars and 14,203 fills.
- Maximum attribution bar residual: 1.04e-11 USDT.
- A separate standard-library/Decimal evidence test recomputes all 798 monthly
  returns, 1,120 overlapping 12/24-month windows, calendar-year returns,
  concentration arithmetic and attribution totals from the 14 raw CSV accounts;
  it also verifies all 64 CSV hashes and frozen source fingerprints.
- Protocol SHA-256: `cfe9ce2290ffaca44593aa385fb5299795ee5f65a57a26fb0d621a3a1c2a4627`.

## Limits

- Adaptive diagnostic on reused dates and three surviving assets; no new holdout or confirmatory significance.
- Rolling windows overlap; observed negative-window fractions are not future loss probabilities.
- Excluding the best three months is ex-post arithmetic, not an executable timing strategy.
- Additive asset contributions describe this funded shared-cash path, not stand-alone asset strategies.
- Omissions use only 10,000 USDT and nominal/joint scenarios; smaller accounts and isolated stresses are untested.
- Model fills, fixed fees, terminal marks, account/EEA execution, taxes, operating costs, custody and quote FX retain prior limitations.
- Historical drawdown/annual-volatility requests do not cap future losses; no post-result policy or universe reselection.

Live eligibility remains false. [Frozen plan](ROBUSTNESS_RESEARCH_PLAN.md).

## Reproduce

```bash
python -m scripts.download_spot_flow
python -m scripts.research_robustness freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_robustness run --jobs 4
```
