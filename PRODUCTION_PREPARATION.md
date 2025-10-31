# Production Preparation Guide for Theta Bot

## Overview

According to IMPLEMENTATION_SUMMARY.txt, the theta bot model has been fully implemented and mathematically verified, but all performance tests (R=0.492, Sharpe=14.24) were conducted only on **synthetic data**. Before deploying the bot for real trading, we need to complete the following minimum steps:

## Minimum Steps Before Trading

### 1. Test on Real Market Data
Run predictions on actual market data (e.g., BTCUSDT_1h.csv) and verify that performance remains good.

### 2. Run Control Tests
Execute permutation and noise tests to confirm the model truly found a signal and isn't overfitting.

### 3. Optimize Hyperparameters
Find the best parameter settings (q, lambda, etc.) specifically for real market data.

## Installation

First, ensure all dependencies are installed:

```bash
pip install numpy pandas scipy matplotlib requests
```

Optional for advanced statistical tests:
```bash
pip install statsmodels
```

## Usage Guide

### Step 1: Download Real Market Data

Download historical data from Binance:

```bash
python download_market_data.py --symbol BTCUSDT --interval 1h --limit 2000
```

Or use an existing CSV file:

```bash
python download_market_data.py --csv path/to/BTCUSDT_1h.csv
```

**Output:** `real_data/BTCUSDT_1h.csv`

### Step 2: Validate Data Quality

Check data quality and run basic statistical tests:

```bash
python validate_real_data.py --csv real_data/BTCUSDT_1h.csv
```

This performs:
- Data quality checks (missing values, outliers)
- Stationarity tests (ADF test)
- Autocorrelation analysis
- Permutation test (control)

**Output:** `validation_output/validation_results.json`

### Step 3: Run Predictions on Real Data

Test the model on real market data:

```bash
python theta_predictor.py --csv real_data/BTCUSDT_1h.csv --window 512 --outdir test_output
```

This will:
- Run walk-forward predictions with no lookahead bias
- Test multiple horizons (h=1,2,4,8,16,32)
- Calculate correlation and hit rate
- Simulate trading with transaction costs

**Output:** 
- `test_output/theta_prediction_metrics.csv`
- `test_output/theta_predictions_h*.csv`
- `test_output/theta_predictions.png`
- `test_output/theta_trade_sim.png`

### Step 4: Run Control Tests

Verify the model distinguishes real signal from noise:

```bash
python theta_horizon_scan_updated.py --csv real_data/BTCUSDT_1h.csv --test-controls --outdir test_output
```

This runs:
- **Permutation test**: Shuffles data to verify predictions fail on random data
- **Noise test**: Adds noise to verify robustness
- **Resonance scan**: Identifies optimal prediction horizons

**Expected:** Control tests should show correlation ≈ 0

**Output:**
- `test_output/theta_resonance.csv`
- `test_output/theta_resonance_permutation.csv`
- `test_output/theta_resonance_noise.csv`

### Step 5: Optimize Hyperparameters

Find the best parameters for your specific market data:

```bash
python optimize_hyperparameters.py --csv real_data/BTCUSDT_1h.csv
```

This will:
- Grid search over key parameters (q, n_terms, n_freqs, lambda)
- Use validation set to prevent overfitting
- Test best parameters on multiple horizons

**Output:**
- `optimization_output/best_hyperparameters.json`
- `optimization_output/optimization_results.csv`
- `optimization_output/optimized_performance.png`

### Step 6: Comprehensive Production Check

Run all tests automatically:

```bash
python production_readiness_check.py --csv real_data/BTCUSDT_1h.csv
```

This executes all validation steps and generates a comprehensive report.

**Output:**
- `production_check/PRODUCTION_READINESS.md` - Summary report
- `production_check/production_readiness_report.json` - Detailed results
- `production_check/validation/` - Validation results
- `production_check/predictions/` - Prediction results
- `production_check/control_tests/` - Control test results
- `production_check/optimization/` - Optimization results

## Evaluation Criteria

### Success Indicators

✓ **Real Data Performance:**
- Correlation r > 0.1 for h=1 or h=2
- Hit rate > 52% (above random)
- Positive Sharpe ratio > 1.0

✓ **Control Tests:**
- Permutation test: r ≈ 0 (< 0.05)
- Noise test: Performance degrades gracefully
- Results significantly better than random

✓ **Statistical Validation:**
- p-value < 0.05 for best horizons
- Consistent performance across validation periods
- No data leakage detected

### Warning Signs

⚠ **Poor Real Data Performance:**
- Correlation r ≈ 0 or negative
- Hit rate ≈ 50% (random guessing)
- Negative cumulative returns

⚠ **Failed Control Tests:**
- High correlation on permuted data
- No performance difference between real and shuffled data
- Suggests overfitting or data leakage

⚠ **Optimization Issues:**
- No parameter combination works well
- Unstable performance across different parameter sets
- Suggests model may not capture real market dynamics

## Parameter Reference

### Hyperparameters to Optimize

- **q** (0.3-0.7): Modular parameter, controls theta function behavior
- **n_terms** (8-16): Number of theta series terms
- **n_freqs** (4-8): Number of frequency components
- **lambda** (0.1-10.0): Ridge regression regularization

### Recommended Default Values

Based on synthetic data testing:
- q = 0.5
- n_terms = 16
- n_freqs = 8
- lambda = 1.0
- window = 512

**Note:** These may need adjustment for real market data!

## Comparison: Synthetic vs Real Data

### Expected Differences

| Metric | Synthetic Data | Real Market Data (Expected) |
|--------|----------------|------------------------------|
| Correlation (h=1) | 0.492 | 0.05-0.20 |
| Hit Rate (h=1) | 65.6% | 52-58% |
| Sharpe Ratio | 14.24 | 1.0-3.0 |
| Resonance Peak | h=1 clear | h=1,2,4 possible |

Real market data is typically:
- **Noisier**: Lower correlations expected
- **Less predictable**: Lower hit rates expected
- **More complex**: May require different parameters

## Next Steps After Validation

If all tests pass successfully:

1. **Paper Trading**
   - Deploy bot in paper trading mode
   - Monitor performance for 1-2 weeks
   - Compare predictions to actual market moves

2. **Risk Management**
   - Set position size limits
   - Implement stop-loss mechanisms
   - Define maximum drawdown thresholds

3. **Live Monitoring**
   - Track prediction accuracy daily
   - Monitor for model drift
   - Be ready to disable bot if performance degrades

4. **Gradual Deployment**
   - Start with small position sizes
   - Increase gradually as confidence grows
   - Keep detailed performance logs

## Troubleshooting

### "Module not found" errors

```bash
pip install numpy pandas scipy matplotlib requests
```

### "Insufficient samples" warning

Need at least 1000 samples (better 2000+) for reliable testing.

### Control tests show high correlation

This suggests potential overfitting or data leakage. Review:
- Window size (might be too large)
- Feature generation (check for lookahead)
- Ridge regularization (might be too weak)

### Real data performance is much worse than synthetic

This is normal! Synthetic data is idealized. If real data shows r>0.05 and hit>51%, that's already meaningful.

### Optimization takes too long

Use `--skip-optimization` flag for quick testing, or reduce the parameter grid in `optimize_hyperparameters.py`.

## File Structure

```
theta-bot/
├── download_market_data.py          # Step 1: Get real data
├── validate_real_data.py            # Step 2: Validate data
├── theta_predictor.py               # Step 3: Run predictions
├── theta_horizon_scan_updated.py    # Step 4: Control tests
├── optimize_hyperparameters.py      # Step 5: Optimize params
├── production_readiness_check.py    # Step 6: Full validation
├── PRODUCTION_PREPARATION.md        # This file
├── real_data/                       # Downloaded market data
├── validation_output/               # Validation results
├── test_output/                     # Prediction results
├── optimization_output/             # Optimization results
└── production_check/                # Full check results
```

## References

- **IMPLEMENTATION_SUMMARY.txt** - Original implementation details
- **EXPERIMENT_REPORT.md** - Synthetic data test results
- **CTT_README.md** - Technical documentation
- **COPILOT_BRIEF_v2.md** - Original specification

## Support

For issues or questions:
1. Check IMPLEMENTATION_SUMMARY.txt for implementation details
2. Review EXPERIMENT_REPORT.md for expected behavior
3. Examine output logs and error messages
4. Verify data quality and format

---

**IMPORTANT:** This is research software. Always test thoroughly with paper trading before deploying with real money. Past performance (especially on synthetic data) does not guarantee future results.
