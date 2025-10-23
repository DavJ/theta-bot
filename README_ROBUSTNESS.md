# Robustness Suite

## Usage
```bash
python robustness_suite.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --pred-ensemble avg --max-by transform \
  --dense-np 16 \
  --out robustness_report.csv
```

### What to expect
- `corr_base` should match the "corr_pred_true" summary you see now (~0.75).
- `corr_lag1`, `corr_shuffle_true`, `corr_shuffle_pred` **must be ~0** if there's no leakage.
- `idx_ok` should be True and `time_monotonic` should be True.
- If `--dense-np` is set, you'll get `corr_dense` and `delta_corr_dense` (improvement from denser psi grid).
