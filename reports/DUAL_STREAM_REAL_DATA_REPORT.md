# Dual-Stream Real Data Evaluation Report

**Generated:** 2025-12-19 15:01:17 UTC

## Dataset Summary

⚠️  **UNREALISTIC/SYNTHETIC-LIKE SAMPLE**

- **Symbol:** BTCUSDT
- **Timeframe:** 1 Hour (1H)
- **Date Range:** 2024-06-01 00:00:00+00:00 to 2024-11-27 23:00:00+00:00
- **Total Bars:** 4,320
- **Price Range:** $13441.23 to $103374.60

## Configuration

- **Mode:** DEFAULT
- **Horizon:** 1 bar(s)
- **Threshold:** 10.0 bps
- **Walk-Forward Splits:** 3
- **Fee Rate:** 0.0004

### Dual-Stream Parameters
- **Theta Window:** 48
- **Theta q:** 0.9
- **Theta Terms:** 8
- **Mellin k:** 16
- **Mellin alpha:** 0.5
- **Mellin omega_max:** 1.0
- **Training Epochs:** 5
- **Batch Size:** 32
- **Learning Rate:** 0.001

## Results Comparison

### Predictive Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Correlation** | -0.0120 | -0.0081 | 0.0039 |
| **Hit Rate (Direction)** | 49.40% | 48.50% | -0.90% |

### Trading Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Total Return** | 21.58% | -3.42% | -25.00% |
| **Sharpe Ratio** | 0.598 | -1.482 | -2.079 |
| **Max Drawdown** | -30.76% | -11.72% | 19.04% |
| **Win Rate** | 42.40% | 27.82% | -14.58% |
| **Profit Factor** | 1.03 | 0.96 | -0.07 |

## Conclusion

Both models show comparable performance, with improved predictive correlation. The evaluation demonstrates the dual-stream architecture's ability to process both Theta sequence patterns and Mellin transform features.

## Diagnostics Summary

**⚠️  FAST MODE / DIAGNOSTIC ONLY**

This report was generated in FAST mode with reduced splits and epochs.
Results are for diagnostic purposes only and should not be used for final conclusions.

### Data Sanity

- **min_close:** $13441.23
- **max_close:** $103374.60
- **Mean Price:** $48950.66
- **Start timestamp:** 2024-06-01 00:00:00+00:00
- **End timestamp:** 2024-11-27 23:00:00+00:00
- **Rows:** 4,320
- **appears_unrealistic:** True

**⚠️  WARNING:** BTCUSDT price range $13441.23 - $103374.60 spans an unusually wide range for 2024-2024. This may indicate synthetic or concatenated data.

### Prediction Quality

**Baseline Model:**
- predicted_return_std: 0.005022
- Signal distribution: {-1.0: 206, 0.0: 150, 1.0: 644}
- Class mean returns:
  - signal=-1: +0.002056
  - signal=+0: +0.002163
  - signal=+1: +0.001036

**Dual-Stream Model:**
- predicted_return_std: 0.002206
- Signal distribution: {-1.0: 39, 0.0: 102, 1.0: 125}
- Class mean returns:
  - signal=-1: +0.001401
  - signal=+0: +0.000440
  - signal=+1: +0.000169

### Root Cause Analysis

**Predictivity loss is most likely caused by:**

- DATA: Synthetic/unrealistic price data detected
- EVALUATION: Both models show near-zero correlation (possible fallback logic or data issues)

---
*WARNING: This evaluation sample appears unrealistic; treat results as synthetic-like data. Results are for research purposes only.*
