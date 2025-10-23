
biquat_max_patch — ZIP patch bundle
===================================

Co to přidá:
  • Nové přepínače do theta_eval_hbatch_biquat.py
      --pred-ensemble {avg,max}   (default: avg)
      --max-by {contrib,transform} (default: contrib)

  • Nové skládání predikce:
      avg  -> původní součet přes ψ (robustní)
      max  -> na každém kroku vezme dominantní ψ
               - max-by=contrib   vybírá dle |X_j * β_j| (doporučeno)
               - max-by=transform vybírá dle |X_j| a predikce je X_j * β_j

Jak použít (doporučené):
------------------------
  1) Udělej si branch a zálohu souboru:
       git checkout -b biquat-max
       cp theta_eval_hbatch_biquat.py theta_eval_hbatch_biquat.py.bak

  2) Spusť patcher:
       python apply_patch.py theta_eval_hbatch_biquat.py

  3) Ověř diff:
       git --no-pager diff -U3 theta_eval_hbatch_biquat.py

  4) Spusť test:
       python theta_eval_hbatch_biquat.py              --symbols /path/to/mydata.csv              --interval 1h --window 192 --horizon 4              --minP 24 --maxP 480 --nP 8              --sigma 0.6 --lambda 1e-2 --limit 1000              --phase biquat              --pred-ensemble max --max-by contrib              --out biquat_max_contrib.csv

Poznámky:
---------
  • Patcher je idempotentní – když změny již existují, nechá je být.
  • Pokud by regex nenašel místo pro nahrazení, patcher vypíše varování
    a vytvoří záložní soubor *.bak, nic nezničí.
  • Součástí je i referenční patch_biquat_max.diff (jen pro čtení).

Návrat zpět:
------------
  cp theta_eval_hbatch_biquat.py.bak theta_eval_hbatch_biquat.py
  # nebo git checkout -- theta_eval_hbatch_biquat.py

