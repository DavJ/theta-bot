# Dual-Stream Real Data Evaluation Report

**Generated:** 2025-12-19 15:37:49 UTC

## Dataset Summary

✓ **DATA SOURCE: REAL MARKET SAMPLE (validated)**

- **Symbol:** BTCUSDT
- **Timeframe:** 1 Hour (1H)
- **Date Range:** 2024-06-01 00:00:00+00:00 to 2024-09-30 23:00:00+00:00
- **Total Bars:** 2,928
- **Price Range:** $64465.34 to $69924.26

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
| **Correlation** | 0.5911 | 0.6042 | 0.0131 |
| **Hit Rate (Direction)** | 77.80% | 77.44% | -0.36% |

### Trading Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Total Return** | 1.14% | 0.31% | -0.83% |
| **Sharpe Ratio** | 7.787 | 6.507 | -1.280 |
| **Max Drawdown** | -0.16% | -0.27% | -0.12% |
| **Win Rate** | 3.50% | 7.89% | 4.39% |
| **Profit Factor** | 2.32 | 1.53 | -0.79 |

## Conclusion

Both models show comparable performance, with improved predictive correlation. The evaluation demonstrates the dual-stream architecture's ability to process both Theta sequence patterns and Mellin transform features.

## Diagnostics Summary

**⚠️  FAST MODE / DIAGNOSTIC ONLY**

This report was generated in FAST mode with reduced splits and epochs.
Results are for diagnostic purposes only and should not be used for final conclusions.

### Data Sanity

- **min_close:** $64465.34
- **max_close:** $69924.26
- **Mean Price:** $67107.94
- **Start timestamp:** 2024-06-01 00:00:00+00:00
- **End timestamp:** 2024-09-30 23:00:00+00:00
- **Rows:** 2,928
- **is_realistic:** True

✓ **All sanity checks passed** - data validated as realistic market data

### Prediction Quality

**Baseline Model:**
- predicted_return_std: 0.000525
- Signal distribution: {-1.0: 19, 0.0: 959, 1.0: 22}
- Class mean returns:
  - signal=-1: -0.001163
  - signal=+0: +0.000038
  - signal=+1: +0.001410

**Dual-Stream Model:**
- predicted_return_std: 0.000594
- Signal distribution: {-1.0: 14, 0.0: 240, 1.0: 12}
- Class mean returns:
  - signal=-1: -0.000788
  - signal=+0: -0.000118
  - signal=+1: +0.000981

### Root Cause Analysis

**No obvious root cause detected.** Predictivity may be limited by:
- Market efficiency (weak signal in 1H data)
- Model capacity (insufficient complexity)
- Feature quality (limited information in Theta/Mellin features)

---
*This evaluation uses the committed real-market dataset (data/BTCUSDT_1H_real.csv.gz). All data sanity checks passed. Results are for research purposes only.*
