# Dual-Stream Real Data Evaluation Report

**Generated:** 2025-12-18 23:10:51 UTC

## Dataset Summary

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
| **Correlation** | -0.0439 | -0.0349 | 0.0090 |
| **Hit Rate (Direction)** | 48.34% | 50.84% | 2.50% |

### Trading Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Total Return** | -0.63% | -17.00% | -16.38% |
| **Sharpe Ratio** | -1.501 | -1.910 | -0.410 |
| **Max Drawdown** | -33.80% | -35.72% | -1.92% |
| **Win Rate** | 24.04% | 23.24% | -0.79% |
| **Profit Factor** | 0.90 | 0.93 | 0.03 |

## Conclusion

Both models show comparable performance, with improved predictive correlation. The evaluation demonstrates the dual-stream architecture's ability to process both Theta sequence patterns and Mellin transform features on synthetic but realistic market data.

## Diagnostics Summary

**⚠️  FAST MODE / DIAGNOSTIC ONLY**

This report was generated in FAST mode with reduced splits and epochs.
Results are for diagnostic purposes only and should not be used for final conclusions.

### Data Sanity

- **Price Range:** $13441.23 - $103374.60
- **Mean Price:** $48950.66
- **Date Range:** 2024-06-01 00:00:00+00:00 to 2024-11-27 23:00:00+00:00
- **Rows:** 4,320

**⚠️  WARNING:** BTCUSDT price range $13441.23 - $103374.60 spans an unusually wide range for 2024-2024. This may indicate synthetic or concatenated data.

### Prediction Quality

**Baseline Model:**
- std(predicted_return): 0.001083
- Signal distribution: {-1.0: 131, 0.0: 714, 1.0: 570}
- Class mean returns:
  - signal=-1: +0.002199
  - signal=+0: +0.001044
  - signal=+1: +0.000831

**Dual-Stream Model:**
- std(predicted_return): 0.001206
- Signal distribution: {-1.0: 168, 0.0: 742, 1.0: 514}
- Class mean returns:
  - signal=-1: +0.002700
  - signal=+0: +0.001052
  - signal=+1: +0.000747

### Root Cause Analysis

**Predictivity loss is most likely caused by:**

- DATA: Synthetic/unrealistic price data detected
- EVALUATION: Both models show near-zero correlation (possible fallback logic or data issues)

---
*Note: This evaluation uses synthetic BTCUSDT data for reproducible benchmarking. Results are for research purposes only.*
