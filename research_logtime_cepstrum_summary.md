# Log-Phase / Cepstrum Volatility Study Summary

## 1. Motivation
- Explore non-directional, scale-aware representations of volatility regimes.
- Address limitations of linear-time, scale-dependent risk indicators that miss cross-scale structure.

## 2. Baseline approach: log-phase concentration C
- Constructed from log-scaled volatility features (e.g., realized volatility).
- Phase defined modulo 1 (toroidal representation) to track regime position without directionality.
- Concentration of phase used as a regime indicator focused on risk state, not price direction.

## 3. Extension: internal phase psi
- Introduced an internal phase via cepstral analysis to capture latent structural dynamics beyond absolute volatility level.
- Combined with C to form C_int (concentration augmented with the internal phase) and an ensemble signal S.

## 4. Linear (FFT) cepstrum results
- Linear-time cepstrum psi_linear is stable but largely redundant with C.
- Ensemble S does not outperform C out-of-sample.
- Conclusion: psi_linear does not add orthogonal information to the baseline indicator.

## 5. Logtime (Mellin-approx) cepstrum
- Applied logarithmic time warping within rolling windows before cepstral computation.
- Cepstrum computed on logtime-resampled data, providing a scale-invariant (Mellin-like) representation.
- Captures structures that manifest similarly across temporal scales.

## 6. Empirical results (key finding)
- Logtime cepstrum psi_logtime shows improved out-of-sample behavior.
- psi_logtime is less redundant with C.
- Ensemble S = C + psi_logtime outperforms C alone in embargoed OOS tests.
- Indicates presence of scale-invariant latent volatility structure.

## 7. Interpretation
- Markets exhibit volatility structures that repeat across different time scales.
- Linear-time representations miss this invariance.
- Scale-aware representations improve regime classification; they are not intended for price prediction.

## 8. Limitations
- No directional price alpha demonstrated.
- Effect size is moderate.
- Suitable as a risk or regime component, not as a standalone trading strategy.

## 9. Practical implications
- Risk regime detection.
- Position sizing or exposure scaling.
- Volatility control layers.
- Inputs to margin or stress frameworks.

## 10. Next steps
- Residualize psi against C.
- Test on dissipative volatility dynamics (vol-of-vol).
- Assess stability across timeframes and assets.

---

Scale-invariant cepstral analysis reveals latent volatility structure that linear-time approaches overlook. By combining log-phase concentration with logtime cepstral features, regime classification improves in a risk-focused, non-directional manner, supporting further research into scale-aware volatility modeling.
