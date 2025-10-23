biquat_max_patch_ready
----------------------

Tento patch přidá podporu pro:
- --pred-ensemble {avg,max}
- --max-by {transform,contrib}

Použití:
    python apply_patch_fixed.py theta_eval_hbatch_biquat.py

Výsledek:
    Přidá CLI argumenty a nahradí blok s `pred_delta = np.mean(comp)`
    logikou, která podporuje výběr maxima podle zvolené metody.

