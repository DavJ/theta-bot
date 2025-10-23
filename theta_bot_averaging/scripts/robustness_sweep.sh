#!/bin/bash
for L in 1e-3 2e-3 5e-3; do
  for NP in 8 16 24; do
    python robustness_suite_v3_oos.py \
      --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
      --interval 1h --window 256 --horizon 4 \
      --minP 24 --maxP 480 --nP $NP \
      --sigma 0.8 --lambda $L --limit 2000 \
      --pred-ensemble avg --max-by transform \
      --out results/robustness_L${L}_NP${NP}.csv
  done
done
