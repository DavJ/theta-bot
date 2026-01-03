# Quantile-Based Signal Generation

## Overview

This document describes the alternative signal generation mode added for evaluation purposes. The quantile-based mode allows testing whether the model can rank trading opportunities, independent of the absolute magnitude of predicted returns.

## Motivation

Traditional threshold-based signals use fixed cutoffs (e.g., ±10 bps). This approach works well when:
- Predicted return magnitudes are well-calibrated
- The model produces consistent scale predictions

However, in early evaluation stages, we want to answer: **"Can the model rank opportunities at all?"**

Quantile-based signals address this by:
- Testing ranking ability separate from magnitude calibration
- Providing consistent signal counts across different folds
- Being robust to prediction scale/bias issues

## Signal Generation Modes

### Threshold Mode (Default)

**Usage:** `signal_mode: "threshold"`

**Logic:**
```python
if predicted_return > threshold_bps / 10000:
    signal = 1  # Long
elif predicted_return < -threshold_bps / 10000:
    signal = -1  # Short
else:
    signal = 0  # Neutral
```

**Characteristics:**
- Fixed cutoffs independent of data distribution
- Signal count varies with prediction quality
- Suitable for production trading with fixed risk parameters
- Default mode for compatibility

**Example:** With `threshold_bps: 10` (0.1% or 10 basis points):
- Long if predicted return > +0.001
- Short if predicted return < -0.001

### Quantile Mode (Evaluation Only)

**Usage:** `signal_mode: "quantile"`

**Logic:**
```python
long_threshold = quantile(predicted_returns, 0.95)   # 95th percentile
short_threshold = quantile(predicted_returns, 0.05)  # 5th percentile

if predicted_return > long_threshold:
    signal = 1  # Long
elif predicted_return < short_threshold:
    signal = -1  # Short
else:
    signal = 0  # Neutral
```

**Characteristics:**
- Adaptive cutoffs based on prediction distribution per fold
- Consistent signal count (~10% total: 5% long, 5% short)
- Tests model's ranking ability
- Robust to prediction scale/bias issues
- **Evaluation only** - not suitable for production

**Example:** In a fold with 100 predictions:
- Top 5 predictions → Long signals
- Bottom 5 predictions → Short signals
- Middle 90 predictions → Neutral

## Usage Examples

### Command Line

```bash
# Evaluate with quantile mode
python evaluate_dual_stream_predictivity.py --signal-mode quantile

# Evaluate with threshold mode (default)
python evaluate_dual_stream_predictivity.py --signal-mode threshold
```

### Configuration File

**Threshold mode (btc_1h.yaml):**
```yaml
data_path: "real_data/BTCUSDT_1h.csv"
horizon: 1
threshold_bps: 10
model_type: "logit"
signal_mode: "threshold"  # Default
# ... other params
```

**Quantile mode (dual_stream_quantile.yaml):**
```yaml
data_path: "real_data/BTCUSDT_1h.csv"
horizon: 1
threshold_bps: 10  # Ignored in quantile mode
model_type: "dual_stream"
signal_mode: "quantile"  # Use quantile-based signals
# ... other params
```

### Run with Config

```bash
python scripts/run_walkforward.py --config configs/dual_stream_quantile.yaml
```

### Python API

```python
from theta_bot_averaging.utils import generate_signals
import pandas as pd

# Threshold mode
signals = generate_signals(
    predicted_return,
    mode="threshold",
    positive_threshold=0.001,  # 10 bps
    negative_threshold=-0.001,
)

# Quantile mode
signals = generate_signals(
    predicted_return,
    mode="quantile",
    quantile_long=0.95,   # 95th percentile
    quantile_short=0.05,  # 5th percentile
)
```

## Interpretation

### When to Use Each Mode

**Threshold Mode:**
- Production trading
- When predicted returns are well-calibrated
- When you need consistent risk parameters
- Default evaluation

**Quantile Mode:**
- Early model evaluation
- Testing ranking ability
- Comparing models with different scales
- Diagnosing calibration issues

### Interpreting Results

**Good ranking ability in quantile mode:**
- Correlation > 0 (positive relationship between prediction and outcome)
- Hit rate > 50% (better than random)
- Positive cumulative return

**If quantile mode succeeds but threshold mode fails:**
- Model has ranking ability but poor calibration
- Consider recalibrating predictions
- Adjust thresholds based on historical distribution

**If both modes fail:**
- Model lacks predictive power
- Revisit feature engineering
- Consider different model architecture

## Implementation Details

### Signal Generation Function

Location: `theta_bot_averaging/utils/signal_generation.py`

```python
def generate_signals(
    predicted_return: pd.Series,
    mode: SignalMode = "threshold",
    positive_threshold: float = 0.0005,
    negative_threshold: float = -0.0005,
    quantile_long: float = 0.95,
    quantile_short: float = 0.05,
) -> pd.Series:
    """Generate trading signals from predicted returns."""
```

### Integration Points

1. **BaselineModel** (`theta_bot_averaging/models/baseline.py`)
   - Added `signal_mode` parameter to `__init__`
   - Updated `predict()` to use `generate_signals()`

2. **DualStreamModel** (`theta_bot_averaging/models/dual_stream.py`)
   - Added `signal_mode` parameter to `__init__`
   - Updated both PyTorch and fallback prediction paths

3. **WalkforwardConfig** (`theta_bot_averaging/validation/walkforward.py`)
   - Added `signal_mode: SignalMode = "threshold"` field
   - Passed to model constructors

4. **Evaluation Scripts**
   - `evaluate_dual_stream_predictivity.py`: Added `--signal-mode` CLI flag
   - Config YAML files: Added `signal_mode` parameter

## Testing

### Unit Tests

Location: `tests/test_signal_generation.py`

Tests cover:
- ✅ Threshold mode with various inputs
- ✅ Quantile mode signal distribution
- ✅ NaN handling
- ✅ Empty series
- ✅ Invalid mode error handling
- ✅ Index preservation
- ✅ Extreme quantiles

Run tests:
```bash
python -m pytest tests/test_signal_generation.py -v
```

### Integration Tests

Verified:
- ✅ Walkforward validation with quantile mode
- ✅ Evaluation script with both modes
- ✅ Config file loading with signal_mode
- ✅ No regressions in existing tests (39 tests pass)

## Example: Comparing Modes

Run the demo script:
```bash
python scripts/example_quantile_signals.py
```

Output shows:
```
THRESHOLD MODE (Fixed 10 bps = 0.001)
  Long (1):     41 ( 41.0%)
  Short (-1):   51 ( 51.0%)
  Neutral (0):   8 (  8.0%)

QUANTILE MODE (95th/5th percentile)
  Long (1):      5 (  5.0%)
  Short (-1):    5 (  5.0%)
  Neutral (0):  90 ( 90.0%)
```

## Limitations

### Quantile Mode

1. **Not for production:** Thresholds change per fold, making risk management difficult
2. **Small sample issues:** With few samples, quantiles may be unstable
3. **Ignores magnitude:** Loses information about predicted return strength
4. **Fold-specific:** Thresholds not comparable across folds

### When Results Differ

If quantile and threshold modes show different performance:

**Quantile better:**
- Model has ranking ability but poor calibration
- Solution: Recalibrate predictions or adjust thresholds

**Threshold better:**
- Predictions are well-calibrated
- Quantile mode's forced signal count may be suboptimal

**Both poor:**
- Model lacks predictive power
- Needs fundamental improvements

## References

- Implementation: `theta_bot_averaging/utils/signal_generation.py`
- Tests: `tests/test_signal_generation.py`
- Example: `scripts/example_quantile_signals.py`
- Configs: `configs/dual_stream_quantile.yaml`
- Documentation: `README.md` (Signal Generation Modes section)

## Summary

Quantile-based signal generation is a valuable evaluation tool that tests model ranking ability independent of calibration. Use it during development to assess whether your model can identify good trading opportunities, even if the predicted magnitudes aren't perfect yet.

For production, always use threshold mode with carefully chosen parameters based on your risk management requirements.
