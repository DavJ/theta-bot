# OOS Patch (robustness_suite_v3_oos.py)

Tento patch přináší **čistý OOS evaluátor** fungující spolehlivě s CSV cestami a aktuálním evaluátorem `theta_eval_hbatch_biquat_max.py`.

## Co dělá
- **CSV-first režim**: Zadané cesty k CSV (např. `../prices/BTCUSDT_1h.csv`) se předají evaluátoru, který vygeneruje soubory `eval_h_*.csv`. Ty se následně načtou a spočítají se OOS metriky (corr, hit-rate).
- **Žádné Binance fetche**: Pokud dáte CSV cesty, script nikdy nevolá burzu.
- **Robustní dohledání výstupních eval CSV**: Umí najít varianty typu `eval_h_BTCUSDT_1hcsv.csv`, `eval_h_BTCUSDT_1HCSV.csv`, atd.
- **OOS split**: Tail (1 - split) je test (např. 0.7 -> posledních 30% řádků).

## Instalace
Umístěte `robustness_suite_v3_oos.py` **do stejného adresáře** jako `theta_eval_hbatch_biquat_max.py`:

```
theta_bot_averaging/
 ├─ theta_eval_hbatch_biquat_max.py
 └─ robustness_suite_v3_oos.py   ← sem
```

## Použití (příklad)
```bash
python robustness_suite_v3_oos.py   --symbols "../prices/BTCUSDT_1h.csv,../prices/ETHUSDT_1h.csv"   --interval 1h --window 256 --horizon 4   --minP 24 --maxP 480 --nP 16   --sigma 0.8 --lam 1e-3   --pred-ensemble avg --max-by transform   --limit 2000   --oos-split 0.7   --out ../theta_trading_addon/results/robustness_report_v3_oos.csv
```

## Výstup
- Vytiskne OOS metriky (corr/hit) pro každý symbol
- Uloží souhrn do CSV na cestu z `--out`

## Poznámky
- Script **nepodporuje** čisté symboly bez CSV v tomto režimu.
- Pokud `theta_eval_hbatch_biquat_max.py` generuje jiný vzor názvu souboru, lze rozšířit seznam kandidátů ve funkci `_find_eval_csv_for`.
