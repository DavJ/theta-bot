# Leak-fix patch kit

Použití:
```bash
# 1) Aplikuj patch na evaluator (upraví filtry, rolling a ground-truth zápis)
python patch_theta_eval_biquat.py biquat_max_standalone/theta_eval_hbatch_biquat_max.py

# 2) Znovu spusť robustness suite
python robustness_suite.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 8 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --pred-ensemble avg --max-by transform \
  --dense-np 16 \
  --out robustness_report.csv
```

Co patch dělá:
- `filtfilt()` → `lfilter()` (kauzální).
- `rolling(center=True)` → `center=False` (kauzální).
- Před uložením per-bar řádku **rekonstruuje** `entry_idx`, `compare_idx`, `last_price`, `future_price`, `true_delta` z **raw close** a vynucuje `compare_idx = entry_idx + horizon`.

Poznámka: Pokud evaluator používá stavové IIR filtry, je vhodné je volat inkrementálně (feed-by-feed). Tento patch se zaměřuje na nejkritičtější a nejběžnější netriviální úniky.
