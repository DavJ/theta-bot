# Benchmark Methods Script - Usage Guide

## Overview

The `spot_bot/benchmark_methods.py` script is a comprehensive benchmarking tool that compares multiple trading bot configurations across multiple trading pairs. It helps identify which parameter combinations perform best across different market conditions.

## Features

- **Multi-pair comparison**: Test configurations across different trading pairs simultaneously
- **Two modes**: Baseline (compare 4 psi_mode variants) or Grid (parameter search)
- **Comprehensive metrics**: Sharpe ratio, volatility, max drawdown, turnover, and more
- **Smart ranking**: Composite score balances risk and return
- **Leaderboards**: Aggregated results by method and by pair
- **Walk-forward validation**: Test robustness across time periods
- **Error resilience**: Individual run failures don't crash the entire benchmark

## Quick Start

### Baseline Mode - Compare PSI Modes

Compare the 4 different psi_mode variants with default parameters:

```bash
python spot_bot/benchmark_methods.py \
    --data-dir data/ohlcv \
    --pairs BTCUSDT,ETHUSDT,BNBUSDT \
    --timeframe 1h \
    --fees 0.0005 \
    --out-dir results/baseline \
    --mode baseline \
    --top 10
```

This will test:
- `psi_mode=cepstrum`
- `psi_mode=complex_cepstrum`
- `psi_mode=mellin_cepstrum`
- `psi_mode=mellin_complex_cepstrum`

### Grid Mode - Parameter Search

Run a small parameter grid for more comprehensive testing:

```bash
python spot_bot/benchmark_methods.py \
    --data-dir data/ohlcv \
    --pairs BTCUSDT,ETHUSDT \
    --timeframe 4h \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --fees 0.001 \
    --slippage-bps 2.0 \
    --out-dir results/grid \
    --mode grid \
    --top 20
```

Grid parameters:
- **FFT modes** (cepstrum, complex_cepstrum): psi_window ∈ [128, 256]
- **Mellin modes**: psi_window, mellin_sigma, psi_min_bin, psi_max_frac, etc.

### Walk-Forward Validation

Test robustness with rolling train/test windows:

```bash
python spot_bot/benchmark_methods.py \
    --data-dir data/ohlcv \
    --pairs BTCUSDT \
    --timeframe 1h \
    --out-dir results/walkforward \
    --mode baseline \
    --walk-forward \
    --train-bars 1000 \
    --test-bars 500 \
    --top 5
```

## Input Data Format

The script expects CSV files in the data directory with this naming convention:
- Format: `{PAIR}_{TIMEFRAME}.csv`
- Examples: `BTCUSDT_1h.csv`, `ETHUSDT_4h.csv`

Each CSV must contain these columns:
- `timestamp`: ISO datetime (e.g., "2023-01-01 00:00:00")
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

Example CSV:
```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,30000.0,30100.0,29900.0,30050.0,1234.56
2023-01-01 01:00:00,30050.0,30200.0,30000.0,30150.0,2345.67
...
```

## Output Files

The script generates three CSV files in the output directory:

### 1. `results_runs.csv`
All individual (pair, method) results with full metrics:
- `pair`: Trading pair name
- `method_name`: Method identifier
- `psi_mode`: PSI mode used
- `params_json`: Full configuration as JSON
- `sharpe`: Sharpe ratio (annualized)
- `max_drawdown`: Maximum drawdown
- `final_return`: Total return
- `volatility`: Volatility (annualized)
- `turnover`: Total turnover
- `trades`: Number of trades
- Plus all config parameters

### 2. `leaderboard_methods.csv`
Aggregated performance by method across all pairs:
- `method_name`: Method identifier
- `score_mean`: Mean composite score
- `score_median`, `score_std`: Score statistics
- `sharpe_mean`, `sharpe_median`: Sharpe statistics
- `max_drawdown_mean`, `max_drawdown_median`: Drawdown statistics
- Similar for other metrics

### 3. `leaderboard_pairs.csv`
Aggregated performance by pair across all methods:
- `pair`: Trading pair
- Same aggregated metrics as methods leaderboard

## Composite Score

Methods are ranked using a composite score that balances multiple objectives:

```
score = sharpe - 0.5 × |max_drawdown| - 0.05 × turnover_norm
```

Where:
- **sharpe**: Higher is better (risk-adjusted return)
- **max_drawdown**: Penalizes large drawdowns (typically negative)
- **turnover_norm**: Normalized turnover (penalizes excessive trading)

This formula ensures we prefer methods with:
- High risk-adjusted returns
- Low drawdowns
- Reasonable trading frequency

## Timeframe Annualization

The script correctly annualizes metrics based on the timeframe:
- `1h`: 24 × 365 = 8,760 periods per year
- `15m`: 4 × 24 × 365 = 35,040 periods per year
- `4h`: 6 × 365 = 2,190 periods per year

## Command-Line Arguments

### Required Arguments
- `--data-dir PATH`: Directory containing OHLCV CSV files
- `--out-dir PATH`: Output directory for results

### Data Selection
- `--pairs PAIR1,PAIR2,...`: Comma-separated pair list (or auto-detect from filenames)
- `--timeframe {1h,15m,4h}`: Timeframe for annualization (default: 1h)
- `--start YYYY-MM-DD`: Filter data start date
- `--end YYYY-MM-DD`: Filter data end date

### Backtest Parameters
- `--fees FLOAT`: Transaction fee rate (default: 0.0005)
- `--slippage-bps FLOAT`: Slippage in basis points (default: 0.0)
- `--initial-equity FLOAT`: Starting equity (default: 1000.0)

### Output Control
- `--top N`: Number of top results to print (default: 10)
- `--mode {baseline,grid}`: Comparison mode (default: baseline)

### Walk-Forward Validation
- `--walk-forward`: Enable walk-forward validation
- `--train-bars INT`: Training window size (default: 1000)
- `--test-bars INT`: Testing window size (default: 500)

## Example Workflow

1. **Prepare your data**: Organize OHLCV CSVs in a directory
2. **Run baseline**: Quick comparison of psi_mode variants
3. **Analyze results**: Review leaderboards and top methods
4. **Run grid**: Deeper search with best-performing modes
5. **Validate**: Use walk-forward to test robustness
6. **Deploy**: Use best configuration in live trading

## Performance Tips

- **Baseline mode**: ~4 configs × pairs (~seconds per pair)
- **Grid mode**: ~36 configs × pairs (~minutes per pair)
- **Walk-forward**: Multiply by number of folds

For faster iteration:
- Test on fewer pairs first
- Use shorter date ranges
- Reduce parameter grid size (modify source code)

## Interpreting Results

### Good Method Characteristics
- **High Sharpe ratio** (> 1.0): Good risk-adjusted returns
- **Low max drawdown** (> -0.2): Limited losses
- **Consistent across pairs**: Works in different markets
- **Low std in scores**: Stable performance

### Warning Signs
- **High turnover**: May be overtrading (transaction costs)
- **High score variance**: Unstable across pairs/folds
- **Errors in results**: Configuration may be invalid

## Troubleshooting

### No pairs found
- Check CSV naming: must be `{PAIR}_{TIMEFRAME}.csv`
- Verify timeframe matches `--timeframe` argument

### All results have errors
- Check CSV format and required columns
- Verify data has enough bars for feature computation
- Check for NaN values in price data

### Low performance (all methods similar)
- Data may be too short or too noisy
- Try different date ranges
- Consider data quality issues

## Advanced Usage

### Custom Parameter Grids

To modify the parameter grid, edit the `generate_grid_configs()` function in the source code. Adjust:
- Parameter ranges
- Grid granularity
- Additional parameters

### Custom Scoring

To change the composite score formula, modify `compute_composite_score()` to adjust:
- Weight of Sharpe ratio
- Drawdown penalty
- Turnover penalty
- Add new metrics

## License

This script is part of the theta-bot repository and follows the same license.
