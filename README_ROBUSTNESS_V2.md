# Robustness Suite v2 (non-overlap)

Použití:
```bash
python robustness_suite_v2.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --pred-ensemble avg --max-by transform \
  --dense-np 16 \
  --out robustness_report_v2.csv
```
Co sleduje navíc:
- `corr_lag_h`: korelace predikce s `true_delta` posunutou o **h** (bez překryvu).
- `ac_true_lag1`, `ac_true_lag_h`: autocorrelation ground truth — referenční baseline.
Leak flag = `|corr_lag_h|` **nesmí** přesahovat přirozenou autocorr `|ac_true_lag_h|` (ani 0.1).
