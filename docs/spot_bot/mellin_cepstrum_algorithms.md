# Mellin Cepstrum Algorithms and Mathematical Background

## Introduction

This document provides a detailed explanation of the Mellin transform-based cepstrum algorithms implemented in theta-bot, including the mathematical foundations and implementation details.

## Mellin Transform

### Definition

The Mellin transform of a function f(x) is defined as:

```
M[f](s) = ∫₀^∞ x^(s-1) f(x) dx
```

where s = σ + iω is a complex variable.

### Discrete Implementation

For a discrete signal x[k], k=1..W, we compute the Mellin transform numerically:

1. **Log-domain mapping**: Transform indices to log-domain
   ```
   u[k] = log(k)  for k = 1, 2, ..., W
   ```

2. **Uniform resampling**: Create uniform grid in u-domain
   ```
   u_grid = linspace(log(1), log(W), grid_n)
   ```

3. **Interpolation**: Resample x onto uniform u-grid
   ```
   x_resampled[i] = interp(u_grid[i], u, x)
   ```

4. **Optional weighting**: Apply exponential weighting
   ```
   x_weighted[i] = x_resampled[i] * exp(σ * u_grid[i])
   ```

5. **FFT**: Compute FFT to obtain Mellin transform
   ```
   X_M[ω] = FFT(x_weighted)
   ```

The parameter σ corresponds to the real part of s and controls frequency emphasis.

## Real Mellin Cepstrum

### Algorithm

The real Mellin cepstrum uses only the magnitude of the Mellin transform:

```
C = IFFT(log(|X_M| + ε))
```

where:
- X_M is the Mellin transform
- ε is a small constant for numerical stability (default: 1e-12)
- log operates element-wise
- IFFT is the inverse FFT

### Properties

- Preserves magnitude information only
- Discards phase information (pseudo-phase from arctangent)
- More robust to noise in some cases
- Similar to real FFT-based cepstrum but in Mellin domain

### Use Cases

- When phase information is unreliable or noisy
- When only spectral envelope is of interest
- Faster than complex cepstrum

## Complex Mellin Cepstrum

### Algorithm

The complex Mellin cepstrum preserves true phase information:

1. **Compute Mellin transform**
   ```
   X_M = MellinTransform(x)
   ```

2. **Extract magnitude and phase**
   ```
   mag = |X_M|
   phase = unwrap(angle(X_M))
   ```

3. **Optional phase detrending**
   ```
   if detrend_phase:
       trend = linear_fit(phase vs bin_index)
       phase = phase - trend
   ```

4. **Construct log spectrum**
   ```
   log_spectrum = log(mag + ε) + i*phase
   ```

5. **Inverse FFT**
   ```
   C = IFFT(log_spectrum)
   ```

### Phase Unwrapping

Phase unwrapping removes 2π discontinuities:

```
phase_unwrapped[0] = phase[0]
for i in 1..N-1:
    diff = phase[i] - phase[i-1]
    if diff > π:
        phase_unwrapped[i] = phase_unwrapped[i-1] + diff - 2π
    elif diff < -π:
        phase_unwrapped[i] = phase_unwrapped[i-1] + diff + 2π
    else:
        phase_unwrapped[i] = phase_unwrapped[i-1] + diff
```

### Phase Detrending

Linear detrending removes the linear trend from unwrapped phase:

```
slope = Σ[(i - mean_i)(phase[i] - mean_phase)] / Σ[(i - mean_i)²]
intercept = mean_phase - slope * mean_i
trend[i] = slope * i + intercept
phase_detrended = phase - trend
```

This helps isolate local phase variations from global trends.

### Properties

- Preserves both magnitude and phase information
- True phase (not pseudo-phase from arctangent)
- More sensitive to signal characteristics
- Can capture phase-based periodicities

### Use Cases

- When phase information is important
- For signals with complex phase relationships
- When maximum information preservation is needed

## Phase Extraction (ψ)

### Band Selection

The phase ψ is extracted from a band of cepstral coefficients:

```
min_bin = max(1, psi_min_bin)
max_bin = min(floor(psi_max_frac * len(C)), len(C) // 2)
band = C[min_bin:max_bin]
```

This band selection focuses on the quefrency range most relevant for the analysis.

### Aggregation Methods

#### Peak Aggregation

Select the bin with maximum magnitude:

```
k* = argmax(|band|)
ψ_angle = angle(band[k*])
ψ = (ψ_angle / 2π) mod 1
```

**Advantages:**
- Simple and fast
- Focuses on dominant component
- Deterministic

**Disadvantages:**
- Sensitive to noise
- May miss distributed information

#### Circular Mean Aggregation

Compute weighted circular mean of all bins:

```
weights = |band|^phase_power
combined = Σ(weights * exp(i*angle(band)))
ψ_angle = angle(combined)
ψ = (ψ_angle / 2π) mod 1
```

**Advantages:**
- More robust to noise
- Incorporates information from multiple bins
- Configurable weighting via phase_power

**Disadvantages:**
- Slightly more expensive
- May blur distinct components

### Phase Power

The `phase_power` parameter controls the weighting in circular mean:

- phase_power = 0: Equal weighting (unweighted circular mean)
- phase_power = 1: Linear weighting by magnitude
- phase_power > 1: Emphasis on larger components
- phase_power < 1: More democratic weighting

## Comparison: FFT vs Mellin Cepstrum

### FFT-Based Cepstrum

- **Domain**: Linear frequency domain
- **Transform**: FFT of original signal
- **Suited for**: Signals with linear frequency characteristics
- **Example**: Constant-Q filters in audio

### Mellin-Based Cepstrum

- **Domain**: Log-frequency domain (scale-invariant)
- **Transform**: FFT of log-resampled signal
- **Suited for**: Signals with exponential/multiplicative characteristics
- **Example**: Financial time series with exponential growth

### Key Differences

1. **Scale invariance**: Mellin is naturally scale-invariant
2. **Frequency resolution**: Mellin has logarithmic frequency resolution
3. **Computational cost**: Similar (both require one FFT)
4. **Sensitivity**: Different to different signal characteristics

## Parameter Selection Guidelines

### Grid Size (`grid_n`)

**Theory**: Determines frequency resolution in Mellin domain.

**Guidelines**:
- **64-128**: Fast, coarse resolution, good for smooth signals
- **256**: Default, balanced for most applications
- **512-1024**: High resolution, better for complex signals
- **Trade-off**: Linear increase in computation time

### Sigma (`sigma`)

**Theory**: Real part of complex frequency variable s = σ + iω.

**Effect**:
- **σ = 0**: No weighting (standard Mellin transform)
- **σ > 0**: Emphasizes higher frequencies (x^σ grows with x)
- **σ < 0**: Emphasizes lower frequencies (x^σ decays with x)

**Guidelines**:
- Start with σ = 0
- Try σ ∈ [-0.5, 0.5] for fine-tuning
- Use σ > 0 for high-frequency emphasis
- Use σ < 0 for low-frequency emphasis

### Epsilon (`eps`)

**Theory**: Prevents log(0) singularities.

**Guidelines**:
- Default: 1e-12 (machine precision for double)
- Increase if numerical issues occur
- Should be much smaller than typical signal values
- Too large: Biases results
- Too small: Numerical instability

### Phase Detrending (`detrend_phase`)

**Theory**: Removes linear phase trend from unwrapped phase.

**Guidelines**:
- **True (default)**: Recommended for most cases
- **False**: When linear trend is meaningful
- Detrending improves stability in practice

### Band Parameters (`psi_min_bin`, `psi_max_frac`)

**Theory**: Define quefrency band for phase extraction.

**Guidelines**:
- **psi_min_bin**: Avoid DC and very low quefrencies
  - Typical: 2-4
  - Higher: Focus on faster periodicities
- **psi_max_frac**: Upper bound as fraction of total
  - Typical: 0.2-0.3
  - Lower: Focus on slower periodicities
  - Higher: Include faster periodicities

### Aggregation Method (`psi_phase_agg`)

**Theory**: How to combine information from multiple bins.

**Guidelines**:
- **"peak"**: When dominant component is clear
  - Faster
  - More sensitive to noise
- **"cmean"**: When signal is complex or noisy
  - More robust
  - Slightly slower
  - Requires tuning phase_power

### Phase Power (`psi_phase_power`)

**Theory**: Weighting exponent in circular mean.

**Guidelines**:
- **0.5**: More democratic, reduces outlier influence
- **1.0**: Linear weighting (default)
- **1.5-2.0**: Strong emphasis on dominant components
- **> 2.0**: Approaches peak aggregation

## Implementation Notes

### Numerical Stability

1. **Log stability**: Always add ε before taking log
2. **Phase wrapping**: Use modulo arithmetic carefully
3. **Edge cases**: Handle empty bands, NaN inputs
4. **Floating-point**: Be aware of precision limits

### Efficiency Considerations

1. **FFT efficiency**: Use power-of-2 sizes when possible
2. **Interpolation**: Linear interpolation is fast and sufficient
3. **Vectorization**: NumPy operations are vectorized
4. **Memory**: Avoid unnecessary copies

### Debugging

The `return_debug=True` option provides:
- `psi_n_star`: Peak bin index
- `psi_c_real`: Real part of cepstrum at peak
- `psi_c_imag`: Imaginary part of cepstrum at peak
- `psi_c_abs`: Magnitude at peak
- `psi_angle_rad`: Angle in radians
- `psi`: Final extracted phase value

## References

### Theoretical Background

1. Mellin transform in signal processing
2. Cepstral analysis and applications
3. Phase unwrapping algorithms
4. Circular statistics

### Related Work

- FFT-based cepstrum analysis
- Constant-Q transforms
- Wavelet transforms
- Time-frequency analysis methods

## Future Enhancements

Potential improvements for future versions:

1. **Multi-resolution Mellin**: Hierarchical analysis at multiple scales
2. **Adaptive parameters**: Automatic parameter selection
3. **GPU acceleration**: For real-time processing
4. **Vectorized rolling**: Efficient sliding window computation
5. **Alternative aggregations**: Median, mode, robust estimators
6. **Quality metrics**: Confidence scores for extracted ψ
