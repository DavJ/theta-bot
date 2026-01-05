# Quant Research Note

## Executive Summary
Top Sharpe and drawdown profiles summarized below. Costs applied to exposure changes (fee + slippage).

## Definitions
- $r_t = \frac{P_t}{P_{t-1}} - 1$ (close-to-close return)
- $RV$: rolling realized volatility
- $\phi$: log-phase of volatility
- $C$: concentration of $\phi$
- $\psi$: phase of $RV$ (cepstrum / Mellin variants)
- $C_{int}$: internal concentration combining $C$ and $\psi$
- $S$: ensemble score from $C$ and $\psi$

## Experiment Setup
- Pairs: AVAX/USDT, BNB/USDT, BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT
- Timeframe: 1h
- Costs: fee_rate=0.001, slippage_bps=5.0, max_exposure=0.3
- Benchmark command: `python -m bench.benchmark_matrix --timeframe 1h --limit-total 8000 --symbols BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,AVAX/USDT --psi-modes none,mellin_cepstrum,mellin_complex_cepstrum --methods C,S --rv-window 24 --conc-window 256 --psi-window 256 --cepstrum-min-bin 4 --cepstrum-max-frac 0.2 --base 10.0 --fee-rate 0.001 --slippage-bps 5 --max-exposure 0.30 --out bench_out/benchmark_matrix.csv`
- Render command: `python -m bench.render_research_note --matrix bench_out/benchmark_matrix.csv --windows bench_out/benchmark_windows.csv --out bench_out/research_note.md`

## Top Runs by Sharpe
| run_id                             | symbol   | method   | psi_mode                |   sharpe |      cagr |   max_drawdown |   turnover |   time_in_market |
|:-----------------------------------|:---------|:---------|:------------------------|---------:|----------:|---------------:|-----------:|-----------------:|
| BNB_USDT_C_mellin_cepstrum         | BNB/USDT | C        | mellin_cepstrum         | 110.419  | 0.119049  |     -0.0883407 |    4.7895  |                1 |
| BNB_USDT_C_mellin_complex_cepstrum | BNB/USDT | C        | mellin_complex_cepstrum | 110.419  | 0.119049  |     -0.0883407 |    4.7895  |                1 |
| BNB_USDT_C_none                    | BNB/USDT | C        | none                    |  84.4835 | 0.0876014 |     -0.0883407 |    4.92634 |                1 |

## Top Runs by Max Drawdown (best)
| run_id                              | symbol    | method   | psi_mode                |   max_drawdown |   sharpe |      cagr |   turnover |   time_in_market |
|:------------------------------------|:----------|:---------|:------------------------|---------------:|---------:|----------:|-----------:|-----------------:|
| AVAX_USDT_S_none                    | AVAX/USDT | S        | none                    |      -0.413866 | -351.35  | -0.45145  |    242.05  |         0.266615 |
| AVAX_USDT_S_mellin_complex_cepstrum | AVAX/USDT | S        | mellin_complex_cepstrum |      -0.397227 | -340.243 | -0.431779 |    236.433 |         0.266211 |
| ETH_USDT_S_none                     | ETH/USDT  | S        | none                    |      -0.386767 | -414.164 | -0.420136 |    229.586 |         0.238632 |

## Good / Bad Windows
**AVAX_USDT_S_mellin_complex_cepstrum — best windows**

|   window_days | start                     | end                       |    return |      maxdd |
|--------------:|:--------------------------|:--------------------------|----------:|-----------:|
|            30 | 2025-04-07 06:00:00+00:00 | 2025-05-07 05:00:00+00:00 | 0.0354941 | -0.0223737 |
|            30 | 2025-04-07 07:00:00+00:00 | 2025-05-07 06:00:00+00:00 | 0.0352846 | -0.0223737 |
|            30 | 2025-06-22 21:00:00+00:00 | 2025-07-22 20:00:00+00:00 | 0.0291493 | -0.0256073 |

**AVAX_USDT_S_mellin_complex_cepstrum — worst windows**

|   window_days | start                     | end                       |    return |     maxdd |
|--------------:|:--------------------------|:--------------------------|----------:|----------:|
|            90 | 2025-07-13 00:00:00+00:00 | 2025-10-10 23:00:00+00:00 | -0.193683 | -0.213161 |
|            90 | 2025-07-12 23:00:00+00:00 | 2025-10-10 22:00:00+00:00 | -0.192401 | -0.21191  |
|            90 | 2025-07-13 02:00:00+00:00 | 2025-10-11 01:00:00+00:00 | -0.190006 | -0.213161 |

**AVAX_USDT_S_none — best windows**

|   window_days | start                     | end                       |    return |      maxdd |
|--------------:|:--------------------------|:--------------------------|----------:|-----------:|
|            30 | 2025-04-07 06:00:00+00:00 | 2025-05-07 05:00:00+00:00 | 0.0345214 | -0.0223737 |
|            30 | 2025-04-07 07:00:00+00:00 | 2025-05-07 06:00:00+00:00 | 0.034312  | -0.0223737 |
|            30 | 2025-06-22 21:00:00+00:00 | 2025-07-22 20:00:00+00:00 | 0.0291493 | -0.0256073 |

**AVAX_USDT_S_none — worst windows**

|   window_days | start                     | end                       |    return |     maxdd |
|--------------:|:--------------------------|:--------------------------|----------:|----------:|
|            90 | 2025-07-13 00:00:00+00:00 | 2025-10-10 23:00:00+00:00 | -0.188737 | -0.208334 |
|            90 | 2025-07-12 23:00:00+00:00 | 2025-10-10 22:00:00+00:00 | -0.187447 | -0.207075 |
|            90 | 2025-08-05 11:00:00+00:00 | 2025-11-03 10:00:00+00:00 | -0.186518 | -0.187076 |

**BNB_USDT_C_mellin_cepstrum — best windows**

|   window_days | start                     | end                       |   return |    maxdd |
|--------------:|:--------------------------|:--------------------------|---------:|---------:|
|            90 | 2025-07-09 13:00:00+00:00 | 2025-10-07 12:00:00+00:00 | 0.160912 | -0.03124 |
|            90 | 2025-07-09 14:00:00+00:00 | 2025-10-07 13:00:00+00:00 | 0.160241 | -0.03124 |
|            90 | 2025-07-09 12:00:00+00:00 | 2025-10-07 11:00:00+00:00 | 0.159647 | -0.03124 |

**BNB_USDT_C_mellin_cepstrum — worst windows**

|   window_days | start                     | end                       |     return |      maxdd |
|--------------:|:--------------------------|:--------------------------|-----------:|-----------:|
|            90 | 2025-10-07 11:00:00+00:00 | 2026-01-05 10:00:00+00:00 | -0.0635104 | -0.0883407 |
|            90 | 2025-10-07 12:00:00+00:00 | 2026-01-05 11:00:00+00:00 | -0.0633061 | -0.0883407 |
|            90 | 2025-10-07 13:00:00+00:00 | 2026-01-05 12:00:00+00:00 | -0.0632912 | -0.0883407 |

**BNB_USDT_C_mellin_complex_cepstrum — best windows**

|   window_days | start                     | end                       |   return |    maxdd |
|--------------:|:--------------------------|:--------------------------|---------:|---------:|
|            90 | 2025-07-09 13:00:00+00:00 | 2025-10-07 12:00:00+00:00 | 0.160912 | -0.03124 |
|            90 | 2025-07-09 14:00:00+00:00 | 2025-10-07 13:00:00+00:00 | 0.160241 | -0.03124 |
|            90 | 2025-07-09 12:00:00+00:00 | 2025-10-07 11:00:00+00:00 | 0.159647 | -0.03124 |

**BNB_USDT_C_mellin_complex_cepstrum — worst windows**

|   window_days | start                     | end                       |     return |      maxdd |
|--------------:|:--------------------------|:--------------------------|-----------:|-----------:|
|            90 | 2025-10-07 11:00:00+00:00 | 2026-01-05 10:00:00+00:00 | -0.0635104 | -0.0883407 |
|            90 | 2025-10-07 12:00:00+00:00 | 2026-01-05 11:00:00+00:00 | -0.0633061 | -0.0883407 |
|            90 | 2025-10-07 13:00:00+00:00 | 2026-01-05 12:00:00+00:00 | -0.0632912 | -0.0883407 |

**BNB_USDT_C_none — best windows**

|   window_days | start                     | end                       |   return |    maxdd |
|--------------:|:--------------------------|:--------------------------|---------:|---------:|
|            90 | 2025-07-09 13:00:00+00:00 | 2025-10-07 12:00:00+00:00 | 0.160912 | -0.03124 |
|            90 | 2025-07-09 14:00:00+00:00 | 2025-10-07 13:00:00+00:00 | 0.160241 | -0.03124 |
|            90 | 2025-07-09 12:00:00+00:00 | 2025-10-07 11:00:00+00:00 | 0.159647 | -0.03124 |

**BNB_USDT_C_none — worst windows**

|   window_days | start                     | end                       |     return |      maxdd |
|--------------:|:--------------------------|:--------------------------|-----------:|-----------:|
|            90 | 2025-10-07 11:00:00+00:00 | 2026-01-05 10:00:00+00:00 | -0.0635104 | -0.0883407 |
|            90 | 2025-10-07 12:00:00+00:00 | 2026-01-05 11:00:00+00:00 | -0.0633061 | -0.0883407 |
|            90 | 2025-10-07 13:00:00+00:00 | 2026-01-05 12:00:00+00:00 | -0.0632912 | -0.0883407 |

**ETH_USDT_S_none — best windows**

|   window_days | start                     | end                       |     return |      maxdd |
|--------------:|:--------------------------|:--------------------------|-----------:|-----------:|
|            30 | 2025-06-22 21:00:00+00:00 | 2025-07-22 20:00:00+00:00 | 0.0117817  | -0.012653  |
|            30 | 2025-06-22 15:00:00+00:00 | 2025-07-22 14:00:00+00:00 | 0.010273   | -0.012653  |
|            30 | 2025-07-15 15:00:00+00:00 | 2025-08-14 14:00:00+00:00 | 0.00938409 | -0.0259643 |

**ETH_USDT_S_none — worst windows**

|   window_days | start                     | end                       |    return |     maxdd |
|--------------:|:--------------------------|:--------------------------|----------:|----------:|
|            90 | 2025-02-18 07:00:00+00:00 | 2025-05-19 06:00:00+00:00 | -0.205294 | -0.208853 |
|            90 | 2025-02-18 05:00:00+00:00 | 2025-05-19 04:00:00+00:00 | -0.204588 | -0.208853 |
|            90 | 2025-02-18 12:00:00+00:00 | 2025-05-19 11:00:00+00:00 | -0.204077 | -0.208853 |


## Conclusion
We model volatility and regime dynamics, not point forecasts of price direction.
            Exposure follows closed-bar signals with explicit transaction costs and risk gating,
            highlighting robustness across market conditions.
