# Theta/Möbius diagnostic — 2026-10-22

**Previously studied history; no fresh holdout. Candidate fixed before payoffs.**

Historical improvement screen: **False**.

The fixed Möbius buy gate reduces full 10,000-USDT net profit from +83.37% for
the guarded momentum control to +39.12%; the full-theta gate gives +42.15%.
Its worst full modeled drawdown is 16.35%, versus 15.03% for control.
The same ranking holds on the separately funded later account and at 1,000 USDT.
These profits share the underlying momentum/risk strategy; they are not returns
of an independently profitable theta or arithmetic forecast strategy.

Later Möbius predictive MSE is worse than both full theta and the trailing mean;
correlations are close to zero. Full theta/plain cosine and Möbius/squarefree
cosine forecasts agree to 1.32e-16 and 4.86e-17, respectively, with identical
buy masks on every source row. The experiment therefore rejects this fixed
buy-filter candidate and supports the algebraic equivalence of the controls.
It does not reject every possible arithmetic model or validate a new winner.

## Funded economic results

Each row uses one continuously funded January 2022–September 2026 account.
Net includes modeled trading costs, before tax, hosting, labor and FX risk.

### Initial 1000 USDT

| Policy | Full nominal net | Full joint net | Worst full DD | Later nominal net |
|---|---:|---:|---:|---:|
| guarded_edge | +84.52% | +71.28% | -15.24% | +31.16% |
| theta_full_gate | +43.56% | +38.84% | -14.72% | +15.11% |
| mobius_gate | +40.47% | +36.20% | -16.60% | +12.50% |
| fundamental_gate | +61.14% | +55.72% | -14.03% | +17.75% |
| mean_gate | +54.66% | +47.85% | -15.10% | +15.39% |

### Initial 10000 USDT

| Policy | Full nominal net | Full joint net | Worst full DD | Later nominal net |
|---|---:|---:|---:|---:|
| guarded_edge | +83.37% | +70.47% | -15.03% | +30.78% |
| theta_full_gate | +42.15% | +37.91% | -14.45% | +15.09% |
| mobius_gate | +39.12% | +34.97% | -16.35% | +12.52% |
| fundamental_gate | +59.84% | +55.44% | -13.81% | +17.73% |
| mean_gate | +53.12% | +46.66% | -14.88% | +15.37% |

## Predictive results: 2025–September 2026

Nonoverlapping six-bar targets within each asset; assets remain dependent.

| Family | Samples | MSE | Mean asset correlation |
|---|---:|---:|---:|
| theta_full | 1911 | 0.00103968 | -0.00961 |
| mobius | 1911 | 0.00104205 | -0.00997 |
| fourier_full | 1911 | 0.00103968 | -0.00961 |
| fourier_squarefree | 1911 | 0.00104205 | -0.00997 |
| fundamental | 1911 | 0.00100638 | -0.02282 |
| mean | 1911 | 0.00093133 | -0.01136 |

## Controls and evidence

Full Fourier reuses theta economics; squarefree Fourier reuses Möbius economics only after
independent forecasts and all buy masks agree. These are exact structural controls, not independent wins.

- 171 official archive checksums rechecked.
- 24 old guarded-edge control paths reproduce.
- 120 distinct paths reconcile cash, inventory and NAV.
- 10 exported full accounts independently audited from CSV.
- The CSV audit covers 104,040 account bars and 9,653 fills; maximum NAV residual
  is 6.63e-10 USDT and covariance-volatility residual is 4.10e-14.
- Local validation: 739 regression tests and 8 legacy metrics tests pass.
- The [source/protocol freeze](https://github.com/DavJ/theta-bot/commit/7ff3fce75db14aed4b32679696c390774c43546d)
  was published before these new payoffs were computed. The dates themselves
  were already studied; this publication does not make them an unseen holdout.
- Protocol SHA-256: `24ec87b0b9b18953edf32e492bab3c7f853d2c136b1b2beb5075684db839dec2`.

## Fixed gates

- accounting: **True**
- lower_later_mse_than_theta_and_mean: **False**
- candidate_drawdowns_within_budget: **True**
- candidate_later_full_positive_all_stresses: **True**
- candidate_beats_control_and_theta_nominal_joint: **False**

## Limitations

- Known historical dates and surviving three-asset universe; no fresh holdout.
- Fixed extrapolated cosine regression, not a separately trained supervised predictor.
- Gate predicts 24h close returns; Monday entries and risk exits determine actual holding times.
- Fourier equivalence prevents a claim of special Möbius/theta predictive information.
- USDT research pairs do not establish executable EEA markets.
- Modeled opens, fees, spread and slippage; no order-book, tax, custody, operating-cost or FX model.
- Historical drawdown and volatility settings do not cap future loss.

Live eligibility remains false. [Frozen plan](MOBIUS_THETA_RESEARCH_PLAN.md).

## Reproduce

```bash
python -m scripts.download_spot_flow
python -m scripts.research_mobius_theta freeze
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.research_mobius_theta run --jobs 4
```
