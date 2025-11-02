# Evaluation Summary

Generated: 2025-11-01 00:33:50

**Configuration:**
- Start Capital: $1000.00
- Taker Fee: 0.100% per side
- Interval: 1h
- Limit: 1000 klines

## Results

| dataset | dataset_type | pair | fee_mode | total_pnl_usdt | end_capital_usdt | avg_monthly_pnl_usdt | corr_pred_true | hit_rate |
|---------|--------------|------|----------|----------------|------------------|----------------------|----------------|----------|
| BTCUSDT_1h_binance                       | binance_live | BTCUSDT  | no_fees    |         -39.03 |           960.97 |               -28.13 |         0.0247 |   0.5271 |
| BTCUSDT_1h_binance                       | binance_live | BTCUSDT  | taker_fee  |        -400.59 |           599.41 |              -288.72 |         0.0247 |   0.5271 |
| ETHUSDT_1h_binance                       | binance_live | ETHUSDT  | no_fees    |        -146.84 |           853.16 |              -105.83 |         0.0293 |   0.5080 |
| ETHUSDT_1h_binance                       | binance_live | ETHUSDT  | taker_fee  |        -478.37 |           521.63 |              -344.77 |         0.0293 |   0.5080 |

## Metric Definitions

- **total_pnl_usdt**: Total profit/loss in USDT
- **end_capital_usdt**: Final capital after trading
- **avg_monthly_pnl_usdt**: Average monthly profit/loss
- **corr_pred_true**: Pearson correlation between predicted and true returns
- **hit_rate**: Fraction of correct direction predictions (zeros count as misses)
