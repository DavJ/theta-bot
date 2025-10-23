# stabilní baseline (kauzální, bez leakage)
python robustness_suite_v2.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 16 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --pred-ensemble avg --max-by transform \
  --out robustness_report_v2_stable.csv

