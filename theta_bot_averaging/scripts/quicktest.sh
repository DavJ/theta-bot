#!/bin/bash
python robustness_suite_v3_oos.py \
  --symbols BTCUSDT \
  --interval 1h --window 256 --horizon 4 \
  --minP 24 --maxP 480 --nP 16 \
  --sigma 0.8 --lambda 1e-3 --limit 1000 \
  --pred-ensemble avg --max-by transform \
  --out results/test_btcusdt.csv
