# Dual-Stream Evaluation: Parameter Comparison Report

**Generated:** 2025-12-18 22:35:00 UTC

This report compares three different evaluation runs of the Dual-Stream model against the Baseline on synthetic BTCUSDT 1H data.

## Evaluation Runs

### 1. Fast Mode (Original PR)
- **Configuration:** Default parameters, 3 splits, 5 epochs
- **Purpose:** Quick CI/testing validation
- **Report:** `DUAL_STREAM_REAL_DATA_REPORT.md`

### 2. Full Mode - Default Parameters
- **Configuration:** Default parameters, 5 splits, 50 epochs
- **Purpose:** Proper evaluation with adequate training
- **Report:** `DUAL_STREAM_FULL_DEFAULT_REPORT.md`

### 3. Full Mode - Optimized Parameters
- **Configuration:** Optimized parameters (θ_window=72, terms=12, mellin_k=20, batch=64, lr=5e-4), 5 splits, 50 epochs
- **Purpose:** Test parameter optimization from previous research
- **Report:** `DUAL_STREAM_OPTIMIZED_REPORT.md`

## Parameter Comparison

| Parameter | Fast/Default | Optimized | Rationale |
|-----------|--------------|-----------|-----------|
| **Theta Window** | 48 | 72 | Capture longer cycles |
| **Theta Terms** | 8 | 12 | Model complex patterns |
| **Mellin k** | 16 | 20 | Better frequency resolution |
| **Batch Size** | 32 | 64 | Training stability |
| **Learning Rate** | 1e-3 | 5e-4 | Convergence stability |
| **Epochs** | 5/50 | 50 | Full convergence |
| **Splits** | 3/5 | 5 | Better validation |

## Results Summary

### Predictive Metrics

| Run | Baseline Corr | DS Corr | Δ | Baseline Hit% | DS Hit% | Δ |
|-----|---------------|---------|---|---------------|---------|---|
| **Fast** | -0.0439 | -0.0349 | +0.0090 | 48.34% | 50.84% | +2.50% |
| **Full Default** | 0.0358 | -0.0211 | -0.0569 | 50.25% | 50.25% | -0.00% |
| **Full Optimized** | 0.0358 | -0.0015 | -0.0373 | 50.25% | 50.00% | -0.25% |

### Trading Metrics

| Run | Baseline Return | DS Return | Δ | Baseline Sharpe | DS Sharpe | Δ |
|-----|-----------------|-----------|---|-----------------|-----------|---|
| **Fast** | -0.63% | -17.00% | -16.38% | -1.501 | -1.910 | -0.410 |
| **Full Default** | 33.86% | -0.86% | -34.72% | 1.089 | -0.239 | -1.328 |
| **Full Optimized** | 33.86% | -24.46% | -58.33% | 1.089 | -3.176 | -4.265 |

## Key Findings

### 1. Baseline Performance Improved with More Data
- Fast mode (50% data): -0.63% return, -1.501 Sharpe
- Full mode (100% data): **33.86% return, 1.089 Sharpe**
- Using all available data significantly improved baseline performance

### 2. Dual-Stream Underperformed on This Synthetic Dataset
All three Dual-Stream configurations underperformed the baseline:
- **Fast:** Slightly better correlation (+0.009) but worse returns (-16.4%)
- **Full Default:** Worse correlation (-0.057), near-zero returns (-0.86%)
- **Full Optimized:** Slightly better correlation than default (-0.0015 vs -0.0211) but worse returns (-24.5%)

### 3. Optimized Parameters Did Not Help
The "optimized" parameters (from previous research on different synthetic data):
- **Did not improve** performance on this dataset
- Actually performed **worse** than default parameters
- Highlights sensitivity to data characteristics

### 4. Dataset Characteristics Matter
This synthetic dataset appears to have different characteristics than the data used in DUAL_STREAM_EVALUATION_REPORT.md:
- That report showed: +41% correlation improvement, +40% Sharpe improvement
- This dataset shows: negative correlation, negative Sharpe
- Suggests the periodic patterns in this synthetic data don't match what Dual-Stream was optimized for

## Interpretation

### Why Did Dual-Stream Underperform?

1. **Data Mismatch**: This synthetic dataset was generated independently and may not have the strong periodic components that Dual-Stream excels at detecting.

2. **Overfitting Risk**: The neural network in Dual-Stream (GRU + gated fusion) may be overfitting to training data, especially with limited samples per fold (~850-3400).

3. **Baseline Simplicity Advantage**: On this relatively simple synthetic data with low signal-to-noise ratio, the simpler logistic regression baseline may generalize better.

4. **Parameter Sensitivity**: The "optimized" parameters were tuned for different data. This highlights that Dual-Stream requires careful hyperparameter tuning for each dataset.

### Comparison to Previous Results

**Previous Research (DUAL_STREAM_EVALUATION_REPORT.md) - Different Synthetic Data:**
- Optimized Dual-Stream: Correlation 0.3942, Sharpe 5.55
- Baseline: Correlation 0.2796, Sharpe 3.97
- Result: **+41% correlation, +40% Sharpe improvement**

**This PR (Current Synthetic Data):**
- Best Dual-Stream: Correlation -0.0015, Sharpe -3.18 (optimized)
- Baseline: Correlation 0.0358, Sharpe 1.09
- Result: **Dual-Stream underperforms significantly**

The stark difference suggests:
- The previous synthetic data had strong periodic patterns ideal for Theta functions
- This synthetic data has weaker or different patterns
- Real-world applicability depends heavily on market characteristics

## Recommendations

### 1. For This Synthetic Dataset
**Use Baseline:** The logistic regression baseline clearly outperforms Dual-Stream on this particular synthetic dataset (33.86% return vs -24.46% for best Dual-Stream).

### 2. For Model Development
- **Dataset-Specific Tuning:** Don't assume "optimized" parameters transfer across datasets
- **Hyperparameter Search:** Run grid search for each new dataset
- **Validation Strategy:** Use proper validation set for parameter tuning
- **Ensemble Approach:** Consider combining both models

### 3. For Real Market Data (Next Steps)
To properly evaluate Dual-Stream on real market data:

1. **Test on Real Binance Data:** 
   - Download actual historical data (requires internet)
   - Run evaluation on real market conditions
   - Expect hit rates around 50-52% (realistic for markets)

2. **Test Longer Horizons:**
   - Try 4H, 8H, 24H predictions
   - Longer horizons may show better results (less noise)

3. **Market-Specific Optimization:**
   - Tune parameters for each market separately
   - BTC may need different settings than ETH

4. **Compare to Research Results:**
   - Previous research showed promising results on different synthetic data
   - Real market validation is essential to confirm practical utility

## Usage Examples

```bash
# Fast mode (for CI/quick testing)
python -m theta_bot_averaging.eval.evaluate_dual_stream_real --fast

# Full evaluation with default parameters
python -m theta_bot_averaging.eval.evaluate_dual_stream_real

# Full evaluation with optimized parameters
python -m theta_bot_averaging.eval.evaluate_dual_stream_real --optimized

# Custom output location
python -m theta_bot_averaging.eval.evaluate_dual_stream_real --optimized --output reports/my_report.md
```

## Conclusion

This evaluation demonstrates that:

1. ✅ **The evaluation framework works correctly** - all three runs completed successfully with proper walk-forward validation
2. ⚠️ **Performance is dataset-dependent** - results vary dramatically based on data characteristics
3. ⚠️ **"Optimized" parameters don't generalize** - what works on one dataset may not work on another
4. ✅ **Baseline is competitive** - simple logistic regression achieved 33.86% return on this synthetic data
5. ❓ **Real market validation needed** - synthetic results don't predict real market performance

The Dual-Stream architecture has shown promise in previous research on different synthetic data, but underperforms on this particular dataset. Real market validation with proper hyperparameter tuning is the next critical step.

---

**Note:** All evaluations use synthetic BTCUSDT data for reproducible offline testing. Results are for research purposes only and do not represent real market performance expectations.
