# OOS (Out-of-Sample) Evaluation — v3

Tento skript hodnotí výkon **pouze na TEST části** (posledních 30 % dat defaultně).
Používá stávající evaluator BIQUAT a je plně kauzální.

## Použití
```bash
python robustness_suite_v3_oos.py --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT \
  --interval 1h --window 256 --horizon 4 --minP 24 --maxP 480 --nP 16 \
  --sigma 0.8 --lambda 1e-3 --limit 2000 \
  --pred-ensemble avg --max-by transform \
  --oos-split 0.7 \
  --out robustness_report_v3_oos.csv
```
**Pozn.:** Toto je "walk-forward OOS" skóre na testu (model se na testu může inkrementálně refitovat, ale vždy kauzálně).  
Pokud přidáš do evaluatoru volbu `--oos-freeze-at`, skript přepne do režimu "frozen-weights OOS" (jedno fitnutí na train, aplikace na test).
