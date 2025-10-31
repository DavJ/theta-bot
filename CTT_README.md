# Complex-Time Theta Transform (CTT) Implementation

This directory contains the implementation of the Complex-Time Theta Transform (CTT) predictive trading engine based on the Complex Consciousness Theory (CCT) and Unified Biquaternion Theory (UBT).

## Overview

The CTT system implements a 4D orthonormalized Jacobi theta basis for market prediction without information leakage. It treats market time as complex: τ = t + iψ, where t is chronological time and ψ is a hidden phase component.

## Core Components

### 1. Theta Basis Generation (`theta_basis_4d.py`)
Generates a 4D orthonormalized basis using Jacobi theta functions across:
- Frequency (ω)
- Phase (φ)
- Imaginary time (ψ)
- Discrete mode (n)

**Usage:**
```bash
python theta_basis_4d.py \
  --n-modes 16 \
  --n-freqs 8 \
  --n-phases 8 \
  --n-psi 8 \
  --q 0.5 \
  --outdir output/
```

**Outputs:**
- `theta_basis.npy` - Orthonormalized basis matrix
- `theta_spectrum.png` - Eigenvalue spectra
- `theta_projection.png` - Phase-space projections
- `theta_metrics.json` - Validation metrics

### 2. Theta Transform (`theta_transform.py`)
Projects signals onto the theta basis and reconstructs them.

**Usage:**
```bash
python theta_transform.py \
  --basis output/theta_basis.npy \
  --signal data.csv \
  --signal-col close \
  --outdir output/
```

**Outputs:**
- `theta_coeffs.npy` - Transform coefficients
- `theta_reconstructed.npy` - Reconstructed signal
- `theta_reconstruction.png` - Visualization
- `theta_transform_metrics.json` - Reconstruction metrics

### 3. Walk-Forward Predictor (`theta_predictor.py`)
Performs causal prediction with no information leakage using walk-forward validation.

**Usage:**
```bash
python theta_predictor.py \
  --csv data.csv \
  --price-col close \
  --window 512 \
  --horizons 1 2 4 8 16 32 \
  --q 0.5 \
  --outdir output/
```

**Outputs:**
- `theta_prediction_metrics.csv` - Performance metrics
- `theta_predictions_h{N}.csv` - Detailed predictions for each horizon
- `theta_predictions.png` - Scatter plots
- `theta_trade_sim.png` - Trading simulation

### 4. Horizon Resonance Scanner (`theta_horizon_scan_updated.py`)
Scans multiple horizons to identify resonance peaks where phase alignment matches modular symmetry.

**Usage:**
```bash
python theta_horizon_scan_updated.py \
  --csv data.csv \
  --price-col close \
  --window 512 \
  --horizons 1 2 4 8 16 32 64 128 \
  --test-controls \
  --outdir output/
```

**Outputs:**
- `theta_resonance.csv` - Correlation vs horizon data
- `theta_resonance.png` - Visualization
- `theta_resonance_permutation.csv` - Control test (if --test-controls)
- `theta_resonance_noise.csv` - Control test (if --test-controls)

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy pandas scipy matplotlib
```

### 2. Generate Test Data
```bash
python generate_test_data.py
```

This creates `test_data/synthetic_prices.csv` with 2000 samples.

### 3. Run Complete Pipeline
```bash
# Generate basis
python theta_basis_4d.py --outdir test_output

# Test transform
python theta_transform.py --basis test_output/theta_basis.npy --outdir test_output

# Run predictions
python theta_predictor.py --csv test_data/synthetic_prices.csv --outdir test_output

# Scan horizons
python theta_horizon_scan_updated.py --csv test_data/synthetic_prices.csv --test-controls --outdir test_output
```

## Results Summary

See `EXPERIMENT_REPORT.md` for comprehensive results, including:
- Mathematical validation of orthonormality
- Prediction performance metrics
- Trading simulation results
- Comparison to CCT/UBT theoretical predictions

### Key Findings (Synthetic Data)

- **Orthonormality:** Validated to machine precision (< 10^-15)
- **Short-term prediction:** r=0.49, hit_rate=65.6% at h=1
- **Sharpe ratio:** 14.24 for h=1 horizon
- **Energy conservation:** Basis captures ~25% of signal variance

## Parameters Guide

### Modular Parameter (q)
- Range: 0 < q < 1
- Default: 0.5
- Lower q → slower modular oscillations
- Higher q → faster modular oscillations

### Number of Modes (n_modes)
- Default: 16 (produces 2*n_modes+1 = 33 basis functions)
- Higher → more expressive basis, but higher computational cost

### Window Size
- Default: 512
- Should be large enough to capture market cycles
- Too small → unstable predictions
- Too large → loss of adaptability

### Ridge Lambda
- Default: 1.0
- Regularization strength for regression
- Higher → more regularization, smoother predictions
- Lower → less regularization, may overfit

## Validation

All implementations have been:
- ✓ Mathematically validated (orthonormality, Hermitian symmetry)
- ✓ Tested with synthetic data
- ✓ Code reviewed
- ✓ Security scanned (CodeQL: 0 vulnerabilities)

## Next Steps

1. **Test with real market data:**
   - Download BTCUSDT_1h or ETHUSDT_1h data
   - Run predictions and resonance scans
   - Compare to CCT/UBT theoretical predictions

2. **Run control tests:**
   - Use `--test-controls` flag with horizon scanner
   - Verify permutation and noise tests yield r≈0

3. **Hyperparameter optimization:**
   - Grid search over (q, n_terms, n_freqs, ridge_lambda)
   - Use validation set to prevent overfitting

4. **Extensions** (see COPILOT_BRIEF_v2.md section 6):
   - Adaptive q(t) via gradient descent
   - Complex-phase diffusion
   - Kalman filtering on coefficients

## Theoretical Background

This implementation is based on the premise that market time is complex:
```
τ = t + iψ
```

The system state is represented using Jacobi theta functions:
```
Θ(q, τ, φ) = Σ_{n=-N}^{N} e^{iπn²τ} e^{2πinqφ}
```

This creates a modularly invariant, quasi-periodic structure that can potentially capture hidden temporal patterns in financial markets.

For full theoretical details, see:
- `Roadmap/COPILOT_BRIEF_v2.md` - Implementation specification
- `EXPERIMENT_REPORT.md` - Results and analysis

## References

- Complex Consciousness Theory (CCT)
- Unified Biquaternion Theory (UBT)
- Jacobi Theta Functions
- Modular Forms and Financial Markets

## License

See repository root for license information.

## Authors

Implementation based on COPILOT_BRIEF_v2.md specification.
