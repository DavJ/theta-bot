# Evaluation Summary

Generated: 2025-10-31 23:04:47

**Configuration:**
- Start Capital: $1000.00
- Taker Fee: 0.100% per side
- Interval: 1h
- Limit: 1000 klines

## Results

| dataset | dataset_type | pair | fee_mode | total_pnl_usdt | end_capital_usdt | avg_monthly_pnl_usdt | corr_pred_true | hit_rate |
|---------|--------------|------|----------|----------------|------------------|----------------------|----------------|----------|
| synthetic_prices.csv                     | synthetic    | SYNTH-PAIR | no_fees    |        4692.97 |          5692.97 |              1690.31 |         0.5213 |   0.6797 |
| synthetic_prices.csv                     | synthetic    | SYNTH-PAIR | taker_fee  |        2001.86 |          3001.86 |               721.03 |         0.5213 |   0.6797 |

## Metric Definitions

- **total_pnl_usdt**: Total profit/loss in USDT
- **end_capital_usdt**: Final capital after trading
- **avg_monthly_pnl_usdt**: Average monthly profit/loss
- **corr_pred_true**: Pearson correlation between predicted and true returns
- **hit_rate**: Fraction of correct direction predictions (zeros count as misses)
