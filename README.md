# Theta Bot - Complex-Time Theta Transform Trading Engine

A predictive trading engine based on Complex Consciousness Theory (CCT) and Unified Biquaternion Theory (UBT), using Jacobi theta functions for market prediction.

## Status

✅ **Implementation Complete** - Model fully implemented and mathematically validated  
🔬 **Testing Infrastructure Ready** - Tools for real market data testing now available  
⏳ **Awaiting Real Data Validation** - Model tested on synthetic data only, real market data testing needed before production

## Quick Start

### For Production Testing (Real Market Data)

**⚠️ IMPORTANT:** The model has been tested on synthetic data only. Before deploying for real trading, you must complete the production preparation steps.

See **[PRODUCTION_PREPARATION.md](PRODUCTION_PREPARATION.md)** for complete guidance.

Quick test with one command:

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib requests

# Test with real data (downloads BTCUSDT automatically)
python quick_start.py --symbol BTCUSDT --interval 1h --limit 2000
```

Or use existing CSV data:

```bash
python quick_start.py --csv path/to/BTCUSDT_1h.csv --quick
```

### Manual Testing Steps

1. **Download real market data:**
   ```bash
   python download_market_data.py --symbol BTCUSDT --interval 1h --limit 2000
   ```

2. **Validate data quality:**
   ```bash
   python validate_real_data.py --csv real_data/BTCUSDT_1h.csv
   ```

3. **Run comprehensive tests on multiple pairs (including PLN):**
   ```bash
   # Full test with all pairs (USDT and PLN)
   python test_biquat_binance_real.py
   
   # Quick test mode (fewer pairs and horizons)
   python test_biquat_binance_real.py --quick
   
   # Skip download and use existing data
   python test_biquat_binance_real.py --skip-download
   ```
   
   This generates comprehensive HTML and Markdown reports in `test_output/`:
   - `comprehensive_report.html` - Interactive HTML report
   - `comprehensive_report.md` - Markdown summary
   
   **Features:**
   - Tests multiple trading pairs: BTC, ETH, BNB, SOL, ADA (with USDT and PLN)
   - Multiple test horizons (1h, 4h, 8h, 24h)
   - Strict walk-forward validation (NO data leaks)
   - Comprehensive performance metrics and visualizations
   - Data leak verification checks

4. **Run predictions:**
   ```bash
   python theta_predictor.py --csv real_data/BTCUSDT_1h.csv --window 512
   ```

5. **Run control tests:**
   ```bash
   python theta_horizon_scan_updated.py --csv real_data/BTCUSDT_1h.csv --test-controls
   ```

6. **Optimize hyperparameters:**
   ```bash
   python optimize_hyperparameters.py --csv real_data/BTCUSDT_1h.csv
   ```

## Performance on Synthetic Data

| Metric | h=1 | h=4 | h=8 |
|--------|-----|-----|-----|
| Correlation | 0.492 | 0.193 | -0.053 |
| Hit Rate | 65.6% | 56.4% | 46.1% |
| Sharpe Ratio | 14.24 | 6.70 | -0.82 |
| p-value | <10⁻⁸⁹ | 0.042 | n.s. |

**Note:** Real market data will likely show lower performance. Correlation r > 0.05 and hit rate > 52% is meaningful for real markets.

## Core Components

### Trading System

- **theta_basis_4d.py** - 4D orthonormalized Jacobi theta basis generation
- **theta_transform.py** - Forward and inverse theta transforms
- **theta_predictor.py** - Walk-forward prediction with no lookahead bias
- **theta_horizon_scan_updated.py** - Resonance scanning and control tests
- **generate_test_data.py** - Synthetic data generation for testing

## Production Preparation Tools

- **download_market_data.py** - Download real market data from Binance
- **validate_real_data.py** - Comprehensive data validation and quality checks
- **optimize_hyperparameters.py** - Grid search for optimal parameters
- **production_readiness_check.py** - Automated end-to-end validation
- **quick_start.py** - One-command testing pipeline
- **test_biquat_corrected.py** - Test corrected biquaternion implementation
- **test_biquat_binance_real.py** - Comprehensive test on real Binance data with multiple pairs
- **tools/eval_metrics.py** - Evaluation script for computing performance metrics (earnings, correlation, hit rate)

## Documentation

- **[PRODUCTION_PREPARATION.md](PRODUCTION_PREPARATION.md)** - Complete guide for preparing bot for production (START HERE!)
- **[IMPLEMENTATION_SUMMARY.txt](IMPLEMENTATION_SUMMARY.txt)** - Implementation details and validation results
- **[EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)** - Detailed synthetic data test results
- **[CTT_README.md](CTT_README.md)** - Technical documentation and theory
- **[TEST_BINANCE_METRICS.md](TEST_BINANCE_METRICS.md)** - Evaluation metrics documentation (earnings, correlation, hit rate)

## Requirements

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
requests (optional, for downloading data)
```

## Architecture

The system implements complex-time dynamics: τ = t + iψ

Where:
- t = chronological time
- ψ = hidden phase component (psychological time)

Using Jacobi theta functions:
```
Θ(q, τ, φ) = Σ e^{iπn²τ} e^{2πinqφ}
```

This creates a modularly invariant, quasi-periodic structure for capturing temporal market patterns.

## Validation Status

✅ Mathematical Properties:
- Orthonormality: error < 10⁻¹⁵ (machine precision)
- Hermitian symmetry: error < 10⁻¹⁸
- Energy conservation: validated
- Eigenvalue spectrum: properly normalized

✅ Code Quality:
- Code review: Passed
- CodeQL security scan: 0 vulnerabilities
- Error handling: Robust
- Documentation: Comprehensive

⏳ Production Testing:
- Real market data testing: **IN PROGRESS**
- Control tests: **PENDING**
- Hyperparameter optimization: **PENDING**
- Paper trading: **NOT STARTED**

## Next Steps

1. ✅ Complete implementation
2. ✅ Validate on synthetic data
3. 🔄 **Test on real market data** (current phase)
4. ⏳ Run control tests (permutation, noise)
5. ⏳ Optimize hyperparameters for real data
6. ⏳ Paper trading validation
7. ⏳ Live deployment with risk management

## Warning

⚠️ **This is research software.** Always:
- Test thoroughly with paper trading first
- Use proper risk management
- Monitor performance continuously
- Be prepared to disable if performance degrades

Past performance (especially on synthetic data) does not guarantee future results.

## License

See repository license file.

## References

- Complex Consciousness Theory (CCT)
- Unified Biquaternion Theory (UBT)
- Jacobi Theta Functions
- Modular Forms and Market Dynamics

### Viewing the Theta Function Evaluation (Atlas Evaluation)  
  
A new evaluation document has been added in `theta_bot_averaging/paper/atlas_evaluation.tex`. This LaTeX paper demonstrates the correctness and physical consistency of the implemented Jacobi theta functions (\theta_1–\theta_4) and the choice of nome \(q\) and imaginary time.  
  
To view the evaluation, compile the LaTeX file with a TeX engine such as `pdflatex`:  
  
```bash  
cd theta_bot_averaging/paper  
pdflatex atlas_evaluation.tex  
```  
  
After compilation, open the resulting `atlas_evaluation.pdf` in your preferred PDF viewer to read the detailed analysis.
