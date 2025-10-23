#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_theta_eval_biquat.py
Automatický "leak-fix" patcher pro theta_eval_hbatch_biquat_max.py

Co dělá:
1) Nahradí zero-phase filtry (`filtfilt`) kauzálním `lfilter`.
2) Přepne všechny pandas rolling s `center=True` na `center=False`.
3) Před uložením per-bar řádku vynutí správné indexy a ground-truth:
   compare_idx = entry_idx + horizon
   future_price = close[compare_idx]
   true_delta = future_price - last_price

Bezpečný je, protože:
- Používá textové záplaty pouze tam, kde našel jasné patterny.
- Pokud vzory nenajde, vypíše varování, ale soubor nemění nekorektně.
"""
import re, sys
from pathlib import Path

def patch_file(path):
    src = Path(path).read_text(encoding="utf-8")
    orig = src

    # 1) filtfilt -> lfilter (import + volání)
    if "filtfilt(" in src:
        # Přidej import lfilter pokud chybí
        if "from scipy.signal import lfilter" not in src and "import scipy.signal as signal" not in src:
            src = src.replace("from scipy.signal import filtfilt",
                              "from scipy.signal import lfilter  # patched (leak-fix)")
        # Nahradí volání
        src = src.replace("filtfilt(", "lfilter(")

    # 2) rolling(..., center=True) -> center=False
    src = re.sub(r"rolling\\(([^\\)]*?)center\\s*=\\s*True([^\\)]*?)\\)",
                 r"rolling(\\1center=False\\2)", src)

    # 3) Vynucení správné konstrukce true/future/indices před uložením řádku
    # Pokus najít místo, kde se zapisuje CSV řádek s políčky time, entry_idx, compare_idx, last_price, pred_price, future_price, pred_delta, true_delta
    # a doplnit kontrolu/rekalkulaci.
    guard_block = r