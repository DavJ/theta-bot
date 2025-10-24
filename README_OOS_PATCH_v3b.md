# OOS Patch v3b — auto correct_pred

Nová verze automaticky **vypočítá sloupec `correct_pred`**, pokud ho `theta_eval_hbatch_biquat_max.py` do CSV nevygeneroval.

## Instalace
Nahraď původní `robustness_suite_v3_oos.py` tímto souborem (ve stejném adresáři jako evaluátor).

## Použití
Stejné jako dříve — příklad:

```bash
python robustness_suite_v3_oos.py   --symbols "../prices/BTCUSDT_1h.csv,../prices/ETHUSDT_1h.csv"   --interval 1h --window 256 --horizon 4   --minP 24 --maxP 480 --nP 16   --sigma 0.8 --lam 1e-3   --pred-ensemble avg --max-by transform   --limit 2000   --oos-split 0.7   --out ../theta_trading_addon/results/robustness_report_v3_oos.csv
```

Teď skript sám spočítá `correct_pred = (sign(pred_delta) == sign(true_delta))`, takže není třeba měnit evaluátor.
