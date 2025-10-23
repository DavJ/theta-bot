#!/bin/bash
python robustness_suite_v3_oos.py \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 \
  --minP 24 --maxP 480 --nP 16 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --pred-ensemble avg --max-by transform \
  --out results/robustness_report_v3_stable.csv
