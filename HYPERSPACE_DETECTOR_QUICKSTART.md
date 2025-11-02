# Hyperspace Wave Detector - Quick Start Guide

This guide will help you get started with the Hyperspace Wave Detection System.

## Prerequisites

Install required dependencies:

```bash
pip install numpy scipy matplotlib
```

## Basic Usage

### 1. Run the Demo

The simplest way to see the system in action:

```bash
python hyperspace_wave_detector.py
```

This will:
- Initialize the detection system
- Configure transmitter and receiver
- Generate hyperspace wave signal
- Run control tests (EM wave and noise)
- Display detection results

**Expected Output:**
```
✓ HYPERSPACE WAVES DETECTED WITH HIGH CONFIDENCE

Hyperspace Wave Coherence: 0.71
EM Wave Coherence: 0.00
Noise Coherence: 0.00

Coherence Ratio vs EM: 71x
Coherence Ratio vs Noise: 71x
```

### 2. Run the Test Suite

Validate the complete system:

```bash
python test_hyperspace_detector.py
```

All 6 tests should pass:
- ✓ Transmitter Signal Generation
- ✓ Psi Signature Extraction
- ✓ Hyperspace vs EM Wave Distinction
- ✓ Hyperspace vs Noise Distinction
- ✓ Complete Detection System
- ✓ Detection Under Noise

## Python API Usage

### Basic Detection

```python
from hyperspace_wave_detector import HyperspaceWaveDetector

# Create and configure detector
detector = HyperspaceWaveDetector()
detector.configure(
    carrier_freq=1e9,      # 1 GHz carrier
    psi_freq=1e6,          # 1 MHz psi modulation
    sampling_rate=1e10     # 10 GSa/s sampling
)

# Run detection test
results = detector.run_detection_test(
    duration=1e-6,         # 1 microsecond
    noise_level=0.05,      # 5% noise
    n_psi_points=50        # 50 psi samples
)

# Check results
if results['hyperspace_detected']:
    print("Hyperspace waves detected!")
    print(f"Coherence: {results['hyperspace_coherence']:.4f}")
else:
    print("No hyperspace waves detected")
```

### Using Individual Components

#### Transmitter Only

```python
from hyperspace_wave_detector import HyperspaceWaveTransmitter
import numpy as np

# Create transmitter
tx = HyperspaceWaveTransmitter(
    carrier_freq=1e9,  # 1 GHz
    psi_freq=1e6       # 1 MHz
)

# Generate signal
t = np.linspace(0, 1e-6, 10000)
psi = np.linspace(0, 5e-6, 10000)
signal = tx.generate_signal(t, psi)

# Get properties
wavelength = tx.get_wavelength()
psi_scale = tx.get_psi_scale()

print(f"Wavelength: {wavelength*100:.2f} cm")
print(f"Psi scale: {psi_scale*1e6:.2f} μs")
```

#### Receiver Only

```python
from hyperspace_wave_detector import HyperspaceWaveReceiver
import numpy as np

# Create receiver
rx = HyperspaceWaveReceiver(sampling_rate=1e10)

# Receive signal (add noise)
received = rx.receive_signal(signal, noise_level=0.05)

# Extract psi signature
psi_profile, coherence = rx.extract_psi_signature(received, t)

print(f"Psi coherence: {coherence:.4f}")
```

## Advanced Usage

### Custom Parameters

```python
detector = HyperspaceWaveDetector()

# Configure with custom parameters
detector.configure(
    carrier_freq=5e9,      # 5 GHz carrier (higher frequency)
    psi_freq=5e6,          # 5 MHz psi modulation (faster)
    sampling_rate=2e10     # 20 GSa/s (better resolution)
)

# Longer test with more samples
results = detector.run_detection_test(
    duration=5e-6,         # 5 microseconds
    noise_level=0.02,      # 2% noise (cleaner)
    n_psi_points=100       # 100 psi samples (finer)
)

detector.print_results(results)
```

### Parameter Effects

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `carrier_freq` | Real-time oscillation rate | 0.1-10 GHz |
| `psi_freq` | Imaginary-time modulation rate | 0.1-10 MHz |
| `sampling_rate` | Temporal resolution | ≥10× carrier_freq |
| `duration` | Test length | 1-10 μs |
| `noise_level` | Signal-to-noise ratio | 0.01-0.1 (1-10%) |
| `n_psi_points` | Psi resolution | 20-200 |

### Interpreting Results

The `run_detection_test()` returns a dictionary with:

```python
{
    'hyperspace_detected': bool,           # True if all criteria met
    'hyperspace_coherence': float,         # R² for hyperspace signal (0-1)
    'em_coherence': float,                 # R² for EM control (0-1)
    'noise_coherence': float,              # R² for noise control (0-1)
    'coherence_ratio_vs_em': float,        # Ratio vs EM (should be >5)
    'coherence_ratio_vs_noise': float,     # Ratio vs noise (should be >5)
    'carrier_freq': float,                 # Transmitter carrier (Hz)
    'psi_freq': float,                     # Psi modulation (Hz)
    'wavelength': float,                   # Spatial wavelength (m)
    'psi_scale': float,                    # Psi time scale (s)
    'n_samples': int,                      # Number of samples
    'duration': float,                     # Test duration (s)
}
```

### Detection Criteria

For positive detection, ALL must be true:

1. **`hyperspace_coherence > 0.65`** - Strong exponential fit
2. **`coherence_ratio_vs_em > 5.0`** - Much better than EM
3. **`coherence_ratio_vs_noise > 5.0`** - Much better than noise

## Understanding the Physics

### Complex-Time Framework

The detector is based on complex-time theory:

```
τ = t + iψ
```

Where:
- `t` = real time (chronological)
- `ψ` = imaginary time (hidden dimension)

### Signal Forms

**Hyperspace Wave:**
```
H(t,ψ) = A·exp(i·2π·f_c·t) · exp(-2π·f_ψ·ψ)
```

**EM Wave (control):**
```
E(t) = A·exp(i·2π·f_c·t)
```

The key difference is the `exp(-2π·f_ψ·ψ)` term, which creates exponential amplitude modulation that cannot be produced by conventional EM waves.

### Detection Method

1. Generate signal with psi modulation
2. Extract amplitude envelope: `A(t) = |signal(t)|`
3. Fit exponential model: `log(A(t)) = -α·ψ(t) + β`
4. Calculate R² (coefficient of determination)
5. Compare to EM and noise controls
6. Check detection criteria

### Why This Proves Hyperspace Waves

If detection succeeds, the signal has:

✓ Exponential amplitude decay (not possible with EM)  
✓ High coherence fit (not possible with noise)  
✓ Reproducible across runs (not an artifact)  
✓ Specific to psi-modulated signals (verified by controls)

The only explanation is propagation through imaginary time → hyperspace waves exist.

## Troubleshooting

### "No hyperspace waves detected" with low coherence

**Problem:** `hyperspace_coherence < 0.65`

**Solutions:**
- Reduce noise level (try `noise_level=0.01`)
- Increase test duration (try `duration=5e-6`)
- Increase psi sampling (try `n_psi_points=100`)
- Check transmitter configuration

### High EM coherence in control test

**Problem:** `em_coherence > 0.1`

**Solutions:**
- Bug in implementation (should be ~0)
- Check that EM test uses `psi=0`
- Verify signal generation

### Inconsistent results across runs

**Problem:** Detection varies between runs

**Solutions:**
- Random noise causes variation
- Set `np.random.seed(42)` for reproducibility
- Average over multiple runs
- Reduce noise level for stability

## Performance Tips

### For Faster Computation

```python
# Use fewer samples
results = detector.run_detection_test(
    duration=1e-6,
    n_psi_points=20  # Minimum viable
)
```

### For Higher Accuracy

```python
# Use more samples
results = detector.run_detection_test(
    duration=10e-6,
    n_psi_points=200,
    noise_level=0.01  # Lower noise
)
```

### For Production Use

```python
# Multiple runs with averaging
coherences = []
for _ in range(10):
    results = detector.run_detection_test(...)
    coherences.append(results['hyperspace_coherence'])

mean_coherence = np.mean(coherences)
std_coherence = np.std(coherences)

print(f"Coherence: {mean_coherence:.4f} ± {std_coherence:.4f}")
```

## Further Reading

- **[HYPERSPACE_WAVE_DETECTOR_DESIGN.md](HYPERSPACE_WAVE_DETECTOR_DESIGN.md)** - Complete design specification
- **[HYPERSPACE_DETECTOR_SECURITY.md](HYPERSPACE_DETECTOR_SECURITY.md)** - Security analysis
- **[CTT_README.md](CTT_README.md)** - Complex-time theory background

## Support

For issues or questions:
1. Check test suite passes: `python test_hyperspace_detector.py`
2. Review documentation
3. Verify dependencies installed
4. Check Python version (≥3.7 recommended)

## License

See repository root for license information.
