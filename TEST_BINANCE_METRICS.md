# Test Binance Metrics Documentation

## Overview

This document defines the evaluation metrics used in `tools/eval_metrics.py` for assessing trading model performance on Binance data.

## Metrics

### 1. Total PnL (total_pnl_usdt)

**Definition:** Total profit or loss in USDT over the evaluation period.

**Calculation:**
```
total_pnl_usdt = end_capital - start_capital
```

**Interpretation:**
- Positive value: Profitable strategy
- Negative value: Losing strategy
- Zero: Break-even

**Example:** If starting with $1000 and ending with $1150, total_pnl_usdt = $150

### 2. End Capital (end_capital_usdt)

**Definition:** Final capital in USDT after all trades and fees.

**Calculation:**
```
end_capital_usdt = capital_after_all_trades_and_fees
```

**Interpretation:**
- Higher is better
- Should be compared against start_capital
- Includes impact of trading fees

**Example:** Starting with $1000, ending with $1150 means end_capital_usdt = $1150

### 3. Average Monthly PnL (avg_monthly_pnl_usdt)

**Definition:** Average profit or loss per month, annualized from total PnL.

**Calculation:**
```
time_span_months = (end_date - start_date) / 30 days
avg_monthly_pnl_usdt = total_pnl_usdt / time_span_months
```

**Interpretation:**
- Positive value: Average monthly profit
- Negative value: Average monthly loss
- Useful for comparing strategies across different time periods
- More stable metric than total PnL for variable-length datasets

**Example:** $150 profit over 3 months → avg_monthly_pnl_usdt = $50/month

### 4. Correlation Pred-True (corr_pred_true)

**Definition:** Pearson correlation coefficient between predicted returns and actual (realized) returns.

**Calculation:**
```
corr_pred_true = pearson_correlation(predicted_returns, true_returns)
```

Where:
- `predicted_returns` = model's predicted price changes
- `true_returns` = actual realized price changes

**Mathematical Formula:**
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

Where:
- xi = predicted return at time i
- yi = true return at time i
- x̄ = mean of predicted returns
- ȳ = mean of true returns

**Range:** -1.0 to +1.0

**Interpretation:**
- **r > 0.5**: Strong positive correlation (excellent)
- **r > 0.3**: Moderate positive correlation (good)
- **r > 0.1**: Weak positive correlation (meaningful for markets)
- **r ≈ 0**: No correlation (predictions have no linear relationship with outcomes)
- **r < 0**: Negative correlation (predictions are inversely related to outcomes)

**Market Context:**
- In financial markets, even r > 0.05 can be meaningful
- High correlations (r > 0.5) are rare in real market data
- Synthetic data often shows higher correlations than real data

**Example:** corr_pred_true = 0.25 means predictions have a weak-to-moderate positive correlation with actual returns

### 5. Hit Rate (hit_rate)

**Definition:** Fraction of predictions where the direction (sign) of the predicted return matches the direction of the actual return.

**Calculation:**
```
hit_rate = correct_direction_predictions / total_predictions
```

**Rules:**
- Count as HIT: sign(predicted_return) == sign(true_return) AND true_return ≠ 0
- Count as MISS: sign(predicted_return) ≠ sign(true_return) OR true_return == 0
- **Important:** Zero true returns count as MISSES (no clear direction means no actionable signal)

**Range:** 0.0 to 1.0 (0% to 100%)

**Interpretation:**
- **hit_rate > 0.60**: Excellent directional accuracy
- **hit_rate > 0.55**: Good directional accuracy
- **hit_rate > 0.52**: Meaningful for trading (better than random)
- **hit_rate ≈ 0.50**: Random/no predictive power
- **hit_rate < 0.50**: Worse than random (inverse trading might help)

**Market Context:**
- In financial markets, hit_rate > 0.52 is considered meaningful
- Professional traders often operate with hit_rate around 0.55-0.60
- High hit_rates (> 0.65) are rare in real market data
- Synthetic data often shows higher hit rates than real data

**Example:** hit_rate = 0.57 means 57% of predictions correctly predicted the direction of price movement

## Fee Modes

### no_fees

Evaluation without transaction costs. Useful for:
- Understanding raw model performance
- Comparing models without fee noise
- Theoretical maximum performance

### taker_fee

Evaluation with taker fees applied to each trade. Default: 0.001 (0.1% per side).

**Fee Application:**
- Buy: Effective price = market_price × (1 + fee_rate)
- Sell: Effective price = market_price × (1 - fee_rate)
- Each trade pays the fee on the transaction amount

**Interpretation:**
- More realistic performance estimate
- Shows impact of trading costs
- Strategies must overcome fees to be profitable

**Example:** With taker_fee = 0.001:
- Buy 1 BTC at $50,000: Pay $50,050 (including $50 fee)
- Sell 1 BTC at $51,000: Receive $50,949 (after $51 fee)
- Net: $899 profit (instead of $1,000 without fees)

## Usage Examples

### Basic Evaluation

```bash
# Evaluate with default settings (1000 USDT start, 0.1% taker fee)
python3 tools/eval_metrics.py --repo-root . --start-capital 1000 --taker-fee 0.001
```

### Custom Configuration

```bash
# Evaluate with 5000 USDT start, 0.075% taker fee
python3 tools/eval_metrics.py --repo-root . --start-capital 5000 --taker-fee 0.00075
```

### Specific Pairs

```bash
# Evaluate only BTC and ETH pairs
python3 tools/eval_metrics.py --repo-root . --pairs BTCUSDT ETHUSDT
```

### Different Intervals

```bash
# Evaluate 15-minute klines
python3 tools/eval_metrics.py --repo-root . --interval 15m --limit 2000
```

## Output Format

The script outputs a Markdown table with the following columns:

| Column | Description |
|--------|-------------|
| dataset | Dataset filename or Binance pair identifier |
| dataset_type | Type: 'real', 'synthetic', or 'binance_live' |
| pair | Trading pair symbol (e.g., BTCUSDT) |
| fee_mode | 'no_fees' or 'taker_fee' |
| total_pnl_usdt | Total profit/loss in USDT |
| end_capital_usdt | Final capital in USDT |
| avg_monthly_pnl_usdt | Average monthly PnL in USDT |
| corr_pred_true | Pearson correlation (-1 to +1) |
| hit_rate | Hit rate (0 to 1) |

### Example Output

```
| dataset                              | dataset_type | pair     | fee_mode   | total_pnl_usdt | end_capital_usdt | avg_monthly_pnl_usdt | corr_pred_true | hit_rate |
|--------------------------------------|--------------|----------|------------|----------------|------------------|----------------------|----------------|----------|
| BTCUSDT_1h_binance                   | binance_live | BTCUSDT  | no_fees    |         125.50 |          1125.50 |                41.83 |         0.1234 |   0.5456 |
| BTCUSDT_1h_binance                   | binance_live | BTCUSDT  | taker_fee  |          98.20 |          1098.20 |                32.73 |         0.1234 |   0.5456 |
| ETHUSDT_1h_binance                   | binance_live | ETHUSDT  | no_fees    |          87.30 |          1087.30 |                29.10 |         0.0987 |   0.5312 |
| ETHUSDT_1h_binance                   | binance_live | ETHUSDT  | taker_fee  |          65.40 |          1065.40 |                21.80 |         0.0987 |   0.5312 |
```

## Thresholds and Benchmarks

### Minimal Viable Performance (Real Market Data)

- **corr_pred_true**: > 0.05 (shows some predictive power)
- **hit_rate**: > 0.52 (better than random)
- **total_pnl_usdt** (with fees): > 0 (profitable after costs)

### Good Performance (Real Market Data)

- **corr_pred_true**: > 0.15
- **hit_rate**: > 0.55
- **avg_monthly_pnl_usdt**: > 5% of capital per month

### Excellent Performance (Real Market Data)

- **corr_pred_true**: > 0.30
- **hit_rate**: > 0.60
- **avg_monthly_pnl_usdt**: > 10% of capital per month

### Warning Signs

- **hit_rate < 0.48**: Worse than random (fundamental problem)
- **corr_pred_true < -0.05**: Inverse relationship (consider flipping signals)
- **Large difference between no_fees and taker_fee PnL**: Trading too frequently

## Data Sources

### Local Files

The script searches for CSV files in:
- `real_data/`
- `history/`
- `test_output/`
- Repository root

Required CSV format:
- Must contain 'close' column with price data
- Optional: 'timestamp' column (generated if missing)
- Optional: 'predicted_return' column (uses momentum if missing)

### Binance API

The script fetches real-time klines from Binance public REST API:
- Endpoint: `https://api.binance.com/api/v3/klines`
- No authentication required
- Rate limits apply (1200 requests per minute)
- Maximum 1000 klines per request

## Technical Notes

### Pearson Correlation Implementation

The Pearson correlation is computed using numpy's corrcoef function:

```python
correlation = np.corrcoef(predicted_returns, true_returns)[0, 1]
```

This implements the standard Pearson correlation formula.

### Hit Rate Implementation

Zero true returns are explicitly counted as misses:

```python
for pred, true_val in zip(predicted_returns, true_returns):
    if true_val == 0:
        # Zero counts as miss (no clear direction)
        continue
    elif np.sign(pred) == np.sign(true_val):
        hits += 1

hit_rate = hits / total_predictions
```

This conservative approach ensures that only clear directional predictions count as hits.

### Trading Simulation

The trading simulation uses a simple momentum-based strategy:
1. Go long when predicted_return > 0
2. Close position when predicted_return ≤ 0
3. Fees applied on both buy and sell
4. No short positions (cash when not long)

This is a baseline strategy for metric computation, not a recommendation for actual trading.

## References

- Binance API Documentation: https://binance-docs.github.io/apidocs/spot/en/
- Pearson Correlation: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
- Hit Rate in Trading: Industry standard metric for directional accuracy

## Version History

- v1.0 (2025-10-31): Initial implementation with earnings, correlation, and hit rate metrics
