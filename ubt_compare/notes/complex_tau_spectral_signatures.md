# Checklist: Measurable Spectral Signatures of a Complex Chronofactor τ = t + iψ

This checklist maps theoretical predictions from the symbolic derivation to
observable quantities available in the theta-bot pipeline.

---

## 1. Eigenvalue Observables — λ_i(t) of O(t) = Θ†Θ

| # | What to check | Signature of complex τ | Real-τ baseline |
|---|---------------|------------------------|-----------------|
| 1.1 | Eigenvalue drift rate $d\lambda_i/dt / \lambda_i$ | Drift present even when generator $G$ is anti-Hermitian (A=0) | Zero drift when A=0 |
| 1.2 | Direction of drift | Can reverse without a sign change of a single real parameter; attributable to $\psi'(t) B$ term | Monotone drift tied to sign of Re(Tr G) |
| 1.3 | Condition number $\kappa(O) = \lambda_{\max}/\lambda_{\min}$ | Sudden spike (transient broadening) even without eigenvalue-level instability | Monotone or stable κ |
| 1.4 | Non-monotone eigenvalue trajectory | Present for non-normal G (pseudospectral transient growth) | Absent for normal G |

---

## 2. Entropy Invariant — S_Θ(t)

| # | What to check | Signature of complex τ |
|---|---------------|------------------------|
| 2.1 | $\partial_t S_\Theta$ sign changes | Can change sign without generator eigenvalue sign change; via $-2k_B\psi'(t)\operatorname{Tr}B$ correction |
| 2.2 | Slope of $S_\Theta(t)$ | $2k_B(\operatorname{Tr}A - \psi'(t)\operatorname{Tr}B)$; $\psi'$-dependent slope rather than constant |
| 2.3 | Amplitude of $S_\Theta$ oscillations | Absent for linear flow (S_Θ is linear in t under real τ); oscillatory envelope indicates nonlinear τ(t) |

---

## 3. Phase Invariant — Σ_Θ(t)

| # | What to check | Signature of complex τ |
|---|---------------|------------------------|
| 3.1 | Dominant frequency $f_0$ in FFT of $\Sigma_\Theta(t)$ | $f_0 = \operatorname{Tr}B / (2\pi k_B)$; present for both real and complex τ |
| 3.2 | Phase–entropy coupling $C_{12}(t) = \dot\Sigma_\Theta / \dot S_\Theta$ | **Time-varying** $C_{12}$ → strong evidence for non-zero $\psi'(t)$ |
| 3.3 | Oscillatory Σ_Θ with monotone S_Θ | Indicates complex-conjugate eigenpairs with $\operatorname{Tr}B \neq 0$ (D3 discriminator) |
| 3.4 | Phase winding $\Delta\Sigma_\Theta$ over closed ψ loop | Non-zero winding $= 2\pi k_B \times$ (zeros of det Θ enclosed) → monodromy |

---

## 4. Discriminators D1–D3 (Summary)

### D1: Phase–Entropy Coupling Coefficient
- **Observable:** $C_{12}(t) = \partial_t\Sigma_\Theta(t) / \partial_t S_\Theta(t)$
- **Real-τ prediction:** constant $= \operatorname{Tr}B / (2\operatorname{Tr}A)$
- **Complex-τ prediction:** $C_{12}(t) = [\operatorname{Tr}B + \psi'(t)\operatorname{Tr}A] / [2(\operatorname{Tr}A - \psi'(t)\operatorname{Tr}B)]$ — time-varying
- **How to test:** Compute $C_{12}$ over rolling windows; test for stationarity (ADF test or visual inspection)

### D2: Spectral Conservation Test
- **Observable:** Eigenvalues $\lambda_i(O(t))$
- **Real-τ prediction:** constant when $A = \operatorname{Herm}(G) = 0$
- **Complex-τ prediction:** drift $\propto -2\psi'(t)\Theta^\dagger B\Theta$ even when $A = 0$
- **How to test:** Identify periods where $A \approx 0$ from spectral scan; check whether $\lambda_i$ drift during those periods

### D3: Mode Pairing / Oscillatory Σ_Θ with Monotone S_Θ
- **Observable:** FFT of $\Sigma_\Theta(t)$; monotonicity of $S_\Theta(t)$
- **Prediction:** Peak at $f_0 = \operatorname{Tr}B/(2\pi k_B)$ in $\hat\Sigma$ while $S_\Theta$ is monotone
- **How to test:** Compute power spectrum of $\Sigma_\Theta$; label monotone $S_\Theta$ windows and check for spectral peak

---

## 5. Pseudospectrum / Non-Normal G Signatures

| # | Observable | Signature |
|---|-----------|-----------|
| 5.1 | $\max_{0\le s\le T}\|e^{sG}\|$ (Kreiss constant bound) | $> 1$ even for Hurwitz G → transient growth |
| 5.2 | ε-pseudospectrum radius | Large pseudo-eigenvalue cloud outside spectrum → high sensitivity |
| 5.3 | Hump in $\|O(t)\|$ (non-monotone) | Direct evidence of non-normal transient |

---

## 6. Mapping to Existing Repo Observables

| Repo quantity | Theoretical link |
|---------------|-----------------|
| `theta_transform` output matrix | Plays the role of Θ(τ) |
| Spectral scan eigenvalue plots (`theta_horizon_scan_updated.py`) | $\lambda_i(O(t))$ — check D2 |
| Log-phase sweep (`btc_log_phase_sweep.py`) | Encodes $\psi$ variation; output is $\Sigma_\Theta$ proxy |
| Confidence/SNR time series | Proxy for $S_\Theta(t)$; use for D1 rolling window test |
| CMB pipeline spectral outputs (if present) | $f_0$ peak test (D3) |

---

## 7. Quick-Reference Formulas

$$S_\Theta(\tau) = 2k_B\bigl(t\operatorname{Tr}A - \psi\operatorname{Tr}B + \operatorname{Re}\log\det\Theta_0\bigr)$$

$$\Sigma_\Theta(\tau) = k_B\bigl(t\operatorname{Tr}B + \psi\operatorname{Tr}A + \operatorname{Im}\log\det\Theta_0\bigr)$$

$$\partial_t O\big|_{\psi(t)} = 2\Theta^\dagger(A - \psi'(t)B)\Theta$$

$$C_{12}(t) = \frac{\operatorname{Tr}B + \psi'(t)\operatorname{Tr}A}{2(\operatorname{Tr}A - \psi'(t)\operatorname{Tr}B)}$$

---

*See `../DERIVATION_P2_complex_chronofactor_spectrum.md` for the full symbolic derivation.*
