# Curated keep list (výchozí návrh)

## Důležité adresáře
- `theta_bot_averaging/` – aktuální implementace predikčního jádra (averaging varianta)
- `src/` – pokud existuje modulární zdrojáky

## Důležité skripty (ponechat v kořeni)
- `theta_eval_hbatch_biquat_max.py` – batche a metriky (corr/hit)
- `robustness_suite_v3_oos.py`, `robustness_suite_v2.py`, `robustness_suite.py` – testy robustnosti a OOS
- `biquat_prepare_and_compare.py` – příprava dat + porovnání
- `theta_eval_hstrategy.py`, `theta_eval_hbatch.py` – hold vs prediktor baseline
- `theta_predictor.py`, `theta_backtest.py`, `theta_batch.py` – rychlé predikce / backtest / multisimbol
- `make_prices_csv.py` – export binance dat do CSV
- `oos_pnl.py` – čistý OOS PnL výpočet

## Data / výstupy
- Všechny velké CSV/JSON/PNG/PDF ignorujeme verzí. Pro malé **regression fixtures** používej složku `regression/**`.