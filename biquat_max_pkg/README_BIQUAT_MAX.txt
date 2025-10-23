BIQUAT MAX — standalone evaluator (zip build)
=============================================

Co je uvnitř
------------
- `theta_eval_hbatch_biquat_max.py` — samostatný evaluátor s podporou `--pred-ensemble {avg,max}`
- `scripts/csv_preclean.py` — rychlá předčistka CSV (přejmenování sloupců, seřazení, UTC)
- `scripts/sanity_check.py` — kontrola dat (časové mezery, statistika návratností)

Důležité
--------
Tento evaluátor nijak neupravuje tvoje existující skripty.
Je čistě CSV-based (nevolá API) a používá sinus/cos základ s ridge regresí.

Režimy:
- `--pred-ensemble avg`  : agregace přes všechny periody (robustní baseline)
- `--pred-ensemble max`  : použije jen dominantní periodu v daném čase
  - dominance dle `--max-by transform|contrib`

Příklad:
  python biquat_max_pkg/theta_eval_hbatch_biquat_max.py \
    --symbols /absolutni/cesta/k.csv \
    --interval 1h --window 192 --horizon 4 \
    --minP 24 --maxP 480 --nP 8 \
    --sigma 0.6 --lambda 1e-2 \
    --pred-ensemble max --max-by transform \
    --out summary_max.csv
