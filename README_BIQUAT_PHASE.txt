
BIQUAT PHASE PATCH (ZIP)
========================

Obsah
-----
1) theta/biquat_phase.py — modul s výpočtem bikvaternionové fáze a ortonormální báze
2) README_BIQUAT_PHASE.txt — tento soubor
3) examples/wiring_snippet.txt — hotové úryvky, jak modul připojit

Použití
-------
1) Rozbal do kořene repa:
   unzip biquat_phase_patch.zip -d ~/workspace/theta-bot

2) V místě, kde vytváříte vstupní matici X (např. v theta_eval_hbatch.py) přidejte:
   from theta.biquat_phase import build_biquat_basis

   X_biq, names = build_biquat_basis(prices_win, t_win, sigma=0.8, max_harm=3, ema_spans=(16,32,64), ridge=1e-6, return_names=True)
   X_full = np.hstack([X_existing, X_biq])  # nebo jen X_biq, pokud chcete čistou theta bázi

3) Volitelné CLI parametry viz examples/wiring_snippet.txt.
