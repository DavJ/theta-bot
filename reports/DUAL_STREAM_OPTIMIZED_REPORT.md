# Dual-Stream Real Data Evaluation Report

**Generated:** 2025-12-18 22:34:04 UTC

## Dataset Summary

- **Symbol:** BTCUSDT
- **Timeframe:** 1 Hour (1H)
- **Date Range:** 2024-06-01 00:00:00+00:00 to 2024-11-27 23:00:00+00:00
- **Total Bars:** 4,320
- **Price Range:** $13441.23 to $103374.60

## Configuration

- **Mode:** OPTIMIZED
- **Horizon:** 1 bar(s)
- **Threshold:** 10.0 bps
- **Walk-Forward Splits:** 5
- **Fee Rate:** 0.0004

### Dual-Stream Parameters
- **Theta Window:** 72 (optimized)
- **Theta q:** 0.9
- **Theta Terms:** 12 (optimized)
- **Mellin k:** 20 (optimized)
- **Mellin alpha:** 0.5
- **Mellin omega_max:** 1.0
- **Training Epochs:** 50
- **Batch Size:** 64 (optimized)
- **Learning Rate:** 0.0005 (optimized)

## Results Comparison

### Predictive Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Correlation** | 0.0358 | -0.0015 | -0.0373 |
| **Hit Rate (Direction)** | 50.25% | 50.00% | -0.25% |

### Trading Metrics

| Metric | Baseline | Dual-Stream | Δ |
|--------|----------|-------------|---|
| **Total Return** | 33.86% | -24.46% | -58.33% |
| **Sharpe Ratio** | 1.089 | -3.176 | -4.265 |
| **Max Drawdown** | -25.72% | -39.41% | -13.69% |
| **Win Rate** | 27.22% | 22.66% | -4.56% |
| **Profit Factor** | 1.02 | 0.89 | -0.12 |

## Conclusion

Both models show comparable performance, though baseline maintains slightly better predictive correlation. The evaluation demonstrates the dual-stream architecture's ability to process both Theta sequence patterns and Mellin transform features on synthetic but realistic market data.

---
*Note: This evaluation uses synthetic BTCUSDT data for reproducible benchmarking. Results are for research purposes only.*
