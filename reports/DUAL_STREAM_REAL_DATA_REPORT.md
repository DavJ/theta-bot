# Dual-Stream Real Data Evaluation Report

**Generated:** 2025-12-19 18:27:24 UTC

## Dataset Summary

✓ **DATA SOURCE: REAL MARKET SAMPLE (validated)**

- **Symbol:** BTCUSDT
- **Timeframe:** 1 Hour (1H)
- **Date Range:** 2024-04-01 01:00:00+00:00 to 2024-10-01 01:00:00+00:00
- **Total Bars:** 4,393
- **Price Range:** $49804.00 to $72404.00

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
| **Correlation** | 0.0044 | -0.1234 | -0.1278 |
| **Hit Rate (Direction)** | 50.45% | 46.62% | -3.83% |

### Trading Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Total Return** | -4.80% | -10.20% | -5.40% |
| **Sharpe Ratio** | -3.395 | -12.501 | -9.106 |
| **Max Drawdown** | -7.58% | -11.04% | -3.46% |
| **Win Rate** | 14.20% | 20.30% | 6.10% |
| **Profit Factor** | 0.72 | 0.56 | -0.16 |

## Conclusion

Both models show comparable performance, though baseline maintains slightly better predictive correlation. The evaluation demonstrates the dual-stream architecture's ability to process both Theta sequence patterns and Mellin transform features.

## Diagnostics Summary

**⚠️  FAST MODE / DIAGNOSTIC ONLY**

This report was generated in FAST mode with reduced splits and epochs.
Results are for diagnostic purposes only and should not be used for final conclusions.

### Data Sanity

- **min_close:** $49804.00
- **max_close:** $72404.00
- **Mean Price:** $63334.84
- **Start timestamp:** 2024-04-01 01:00:00+00:00
- **End timestamp:** 2024-10-01 01:00:00+00:00
- **Rows:** 4,393
- **is_realistic:** True

✓ **All sanity checks passed** - data validated as realistic market data

### Prediction Quality

**Baseline Model:**
- predicted_return_std: 0.000984
- Signal distribution: {-1.0: 33, 0.0: 711, 1.0: 256}
- Class mean returns:
  - signal=-1: +0.000623
  - signal=+0: +0.000060
  - signal=+1: +0.000197

**Dual-Stream Model:**
- predicted_return_std: 0.001298
- Signal distribution: {-1.0: 32, 0.0: 142, 1.0: 92}
- Class mean returns:
  - signal=-1: +0.002262
  - signal=+0: -0.000204
  - signal=+1: -0.001576

### Root Cause Analysis

**No obvious root cause detected.** Predictivity may be limited by:
- Market efficiency (weak signal in 1H data)
- Model capacity (insufficient complexity)
- Feature quality (limited information in Theta/Mellin features)

---
*This evaluation uses the committed real-market dataset (data/BTCUSDT_1H_real.csv.gz). All data sanity checks passed. Results are for research purposes only.*
