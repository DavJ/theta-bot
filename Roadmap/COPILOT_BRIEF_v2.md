# COPILOT_BRIEF_v2.md

## Goal
Develop a **predictive trading engine** using the **Complex-Time Theta Transform (CTT)** — a practical implementation derived from the **Complex Consciousness Theory (CCT)** and the **Unified Biquaternion Theory (UBT)**.  
The objective is to predict future market prices without information leakage using an orthonormalized 4D Jacobi theta basis in complex time.

---

## 1. Theoretical foundation

The project assumes that market time is **complex**:
\[
τ = t + iψ
\]
where *t* is chronological time and *ψ* is a hidden phase component (psychological or cognitive time).

The system state is represented as a function:
\[
Θ(q, τ, φ) = Σ_{n=-N}^{N} e^{iπn^2τ} e^{2πinqφ}
\]
where:
- *q* is a modular parameter (typically 0 < q < 1),
- *φ* is a phase variable (potentially complex),
- The basis functions are **Jacobi theta functions** forming a quasi-periodic, fractal, modularly invariant structure.

---

## 2. Implementation targets

### 2.1 Basis generation
Implement a **4D orthonormalized basis**:
- Axes: frequency (ω), phase (φ), imaginary time (ψ), and discrete mode (n).
- Use **complex Gram–Schmidt** orthonormalization over sampled values of Θ(q, τ, φ).
- Verify with:
  - \( Re(O) ≈ I \)
  - \( Im(O) ≈ 0 \)
  - Compute and save eigenvalue spectra before and after normalization.

### 2.2 Transform and inverse
Implement:
- `theta_transform(signal, basis) → coeffs`
- `theta_inverse(coeffs, basis) → reconstructed`
- Validate reconstruction:
  - Correlation (r), RMS error, energy conservation.

### 2.3 No-leak walk-forward prediction
Build `theta_predictor.py`:
- Train window: W samples → predict horizon h.
- Fit model only on `[t−W, t)` and predict at `t+h`.
- Prevent lookahead bias.
- Repeat for horizons `h = {1, 2, 4, 8, 16, 32}`.
- Measure:
  - Pearson correlation between predicted and true deltas.
  - Directional hit rate (% correct sign).
  - Cumulative trading PnL (long/short with transaction cost).

### 2.4 Frequency resonance scan
Implement `theta_horizon_scan.py`:
- Evaluate how predictive correlation varies with horizon.
- Produce plots of correlation vs. horizon.
- Expectation: potential resonance peaks where phase alignment matches modular symmetry.

---

## 3. Diagnostics & outputs

| File | Description |
|------|--------------|
| `theta_basis.npy` | Generated 4D orthonormal basis |
| `theta_coeffs.npy` | Projection coefficients |
| `theta_metrics.json` | Summary of reconstruction & prediction |
| `theta_resonance.csv` | Correlation vs. horizon data |
| `theta_spectrum.png` | Eigenvalue spectra |
| `theta_projection.png` | Toroidal or phase projection of Θ(q, τ, φ) |
| `theta_trade_sim.csv` | Trading simulation summary |

---

## 4. Empirical validation (CCT → UBT)

- The experiment should reproduce the empirical effects observed previously:
  - **Predictive correlation** remains significantly above random baseline.
  - **Permutation test** and **synthetic noise test** yield near-zero predictive value.
  - For real market data (BTCUSDT_1h, ETHUSDT_1h), correlation peaks near horizon ≈ 16–32.
- This confirms that the **theta modular structure encodes quasi-temporal information**, consistent with **CCT/UBT predictions** of complex time dynamics.

---

## 5. Mathematical checks

1. Confirm quasi-orthogonality before GS:
   \[
   O_{ij} = ⟨Θ_i, Θ_j⟩
   \]
   should have small off-diagonal terms.
2. After orthonormalization:
   \[
   O_{ij} = δ_{ij}
   \]
3. Ensure Hermitian symmetry:
   \[
   O = O^†
   \]
4. Compute eigenvalue decomposition before and after normalization to validate transformation stability.

---

## 6. Extensions

- Extend to 4 Jacobi theta functions (Θ₁…Θ₄).
- Introduce **adaptive modular parameter** q(t) learned via gradient descent.
- Explore **complex-phase diffusion**: stochastic drift of ψ.
- Optionally integrate **Kalman filter** on theta coefficients for smoother temporal prediction.

---

## 7. Deliverables

- Code committed under `branch: copilot-task1`
- Tested scripts:
  - `theta_basis_4d.py`
  - `theta_transform.py`
  - `theta_predictor.py`
  - `theta_horizon_scan.py`
- Document results in Markdown (`EXPERIMENT_REPORT.md`) including:
  - Plots
  - Tables
  - Observations on resonance and predictive asymmetry

---

## 8. Summary

The model aims to validate that the **Jacobi-theta-based complex-time decomposition** not only reconstructs past signals but also captures **hidden modular symmetries** enabling **prediction beyond stochastic noise**.

If confirmed, this provides empirical evidence supporting the **CCT/UBT hypothesis** that time evolution in financial (and physical) systems reflects an underlying **complex toroidal metric** with modular invariance.
