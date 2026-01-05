# Scale-Invariant Volatility Regimes — Quant Research Note

## Hypothesis
- Markets exhibit scale-invariant volatility regimes.

## Method
- **Log-phase concentration (C):** phase of log-volatility on the unit circle; rolling concentration gauges regime coherence.
- **Scale-phase concentration (ψ):** logtime/scale cepstrum phase that captures cross-scale structure absent in linear-time views.
- **Ensemble S:** percentile blend of C and ψ (or internal concentration) to gate exposure states.

## Results
- **Pairs that benefit:** High-liquidity majors (e.g., BTC/USDT, ETH/USDT) show steadier drawdowns and fewer OFF/ON flips when ψ_logtime is combined with C; S provides cleaner risk-state separation than C alone.
- **Pairs that do not:** Thin or fast-trending alts with noisy volatility structure gain little; ψ_linear is largely redundant with C.
- **Stability across windows:** ψ_logtime remains stable for conc_window and psi_window in the 128–512 range; very short windows amplify noise, very long windows smear transitions.

## Limitations
- No directional alpha demonstrated.
- Regime-only signal; intended for risk control.

## Practical use
- **Risk gating:** apply S thresholds to OFF/REDUCE exposure when concentration breaks down.
- **Position sizing:** scale target notional by S percentile or risk budget instead of absolute volatility.
- **Overlay:** layer atop existing intents (trend, mean reversion, market making) as a volatility-control component.

## Reproducibility
- Features-only pair scan:
  ```bash
  python -m bench.benchmark_pairs --timeframe 1h --limit-total 8000 --out bench_out/benchmark_pairs.csv --psi-mode complex_cepstrum --psi-window 256 --cepstrum-domain logtime --conc-window 256
  ```
- Strategy evaluation with gating:
  ```bash
  python bench/benchmark_strategies.py --timeframe 1h --limit-total 8000 --out bench_out/strategies.csv --pivot-out bench_out/strategies_pivot.csv --psi-modes scale_phase --kalman-mode meanrev
  ```
- Single pair diagnostic:
  ```bash
  python demo_mellin_cepstrum.py
  ```
