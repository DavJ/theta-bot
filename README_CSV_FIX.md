# Theta CSV Fix – README

Použití:
1) Zkontroluj data:
   python scripts/sanity_check.py /Users/davidjaros/workspace/theta-bot/data/mydata.csv

2) Spusť evaluaci přes wrapper:
   python scripts/run_with_csv_fix.py \
     --symbols /Users/davidjaros/workspace/theta-bot/data/mydata.csv \
     --interval 1h --window 192 --horizon 4 \
     --minP 24 --maxP 480 --nP 8 \
     --sigma 0.6 --lambda 1e-2 --limit 1000 \
     --phase biquat \
     --out tuned_biquat.csv

3) Pokud je `mae_return` pořád mimo, uprav ho v theta_eval_hbatch_biquat.py podle PATCH_NOTES_search_and_replace.txt.
