#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_patch.py
--------------
Regex-based patcher for theta_eval_hbatch_biquat.py

What it does:
  1) Adds CLI flags: --csv-time-col, --csv-close-col
  2) Extends fetch_ohlcv() to accept CSV with custom columns
  3) Tries to inject inverse-transform (normalized delta -> price)

Safe:
  - Writes a timestamped backup to ./backup/
  - If a step can't be done, prints guidance and exits non-destructively.
"""

import re, sys, os, shutil, datetime

TARGET = "theta_eval_hbatch_biquat.py"

ARGPARSE_INJECT = r"""
    parser.add_argument('--csv-time-col', default='time',
                        help='CSV column name for timestamps when --symbols is a CSV file')
    parser.add_argument('--csv-close-col', default='close',
                        help='CSV column name for close prices when --symbols is a CSV file')
"""

FETCH_OHLCV_HINT = r"def fetch_ohlcv(symbol, interval, limit)"

FETCH_OHLCV_PATCH = r'''
def fetch_ohlcv(symbol, interval, limit, csv_time_col="time", csv_close_col="close"):
    """
    If `symbol` looks like a file path (ends with .csv or contains a path sep),
    load CSV with columns [csv_time_col, csv_close_col] and return a df with columns ["time","close"].
    Otherwise, fall back to the existing exchange fetch logic (unchanged).
    """
    import os
    import pandas as pd

    # Heuristic: treat as CSV if a file exists, or if it endswith .csv
    if os.path.exists(symbol) or symbol.lower().endswith(".csv"):
        df = pd.read_csv(symbol)
        if csv_time_col not in df.columns or csv_close_col not in df.columns:
            raise ValueError(f"CSV missing required columns: '{csv_time_col}', '{csv_close_col}'")
        df = df.rename(columns={csv_time_col: "time", csv_close_col: "close"})[["time","close"]]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        return df
    # --- fallback to original implementation below ---
'''

INVERSE_SNIPPET = r'''
        # --- START: inverse-transform normalized delta -> price ---
        import numpy as _np

        # okno posledních W cen (float)
        y_win = close[entry_idx - window: entry_idx].astype(float)
        x = _np.arange(window, dtype=float)

        # lokální trend (intercept + slope*x), bezpečné i bez detrendu
        A = _np.vstack([_np.ones_like(x), x]).T
        try:
            beta_trend, *_ = _np.linalg.lstsq(A, y_win, rcond=None)
            trend_intercept, trend_slope = beta_trend
        except Exception:
            trend_intercept, trend_slope = float(y_win.mean()), 0.0

        mu = float(y_win.mean())
        sigma = float(y_win.std(ddof=0))
        if not _np.isfinite(sigma) or sigma == 0.0:
            sigma = 1.0

        # očekáváme, že y_hat_norm reprezentuje normalizovanou DELTU
        delta_price_hat = float(y_hat_norm) * sigma

        last_price = float(close[entry_idx - 1])

        # trendový přírůstek o 1 krok dopředu (může být 0, pokud detrend nepoužíváš)
        trend_next = trend_intercept + trend_slope * (window)
        trend_curr = trend_intercept + trend_slope * (window - 1)
        trend_delta = float(trend_next - trend_curr)

        pred_price = last_price + delta_price_hat + trend_delta
        if not _np.isfinite(pred_price):
            pred_price = last_price
        # --- END: inverse-transform ---
'''

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def load_text(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def save_text(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def backup(p):
    os.makedirs("backup", exist_ok=True)
    dst = os.path.join("backup", f"{os.path.basename(p)}.{timestamp()}.bak")
    shutil.copy2(p, dst)
    return dst

def inject_argparse_flags(txt):
    # find argparse.ArgumentParser() block
    m = re.search(r"parser\s*=\s*argparse\.ArgumentParser\([^\)]*\)\s*", txt)
    if not m:
        print("[WARN] argparse parser not found; skipping CLI inject.")
        return txt, False

    # Insert our flags after the parser is created
    insert_at = m.end()
    new_txt = txt[:insert_at] + ARGPARSE_INJECT + txt[insert_at:]
    print("[OK] Added --csv-time-col and --csv-close-col flags.")
    return new_txt, True

def patch_fetch_ohlcv(txt):
    # locate def fetch_ohlcv(...) signature
    if FETCH_OHLCV_HINT not in txt:
        print("[WARN] fetch_ohlcv signature not found; skipping CSV loader patch.")
        return txt, False

    # Replace function header with patched block; keep the original fallback by appending original body after the guard
    pat = r"def\s+fetch_ohlcv\s*\(\s*symbol\s*,\s*interval\s*,\s*limit\s*\)\s*:\n"
    if not re.search(pat, txt):
        print("[WARN] fetch_ohlcv def not matched by regex; skipping.")
        return txt, False

    # Rename the original function to keep fallback
    txt2 = re.sub(pat, "def _fetch_ohlcv_exchange(symbol, interval, limit):\n", txt, count=1)

    # Now prepend a new wrapper that handles CSV then calls _fetch_ohlcv_exchange
    wrapper = FETCH_OHLCV_PATCH + "\n" +               "    # if not CSV, use the original exchange fetch:\n" +               "    return _fetch_ohlcv_exchange(symbol, interval, limit)\n"

    # Insert wrapper right before the renamed function definition
    insert_pos = txt2.find("def _fetch_ohlcv_exchange(")
    if insert_pos == -1:
        print("[WARN] could not insert CSV wrapper; skipping.")
        return txt, False

    new_txt = txt2[:insert_pos] + wrapper + "\n" + txt2[insert_pos:]
    print("[OK] Patched fetch_ohlcv with CSV support.")
    return new_txt, True

def try_inject_inverse(txt):
    # Heuristic: find spot where pred_price is used/assigned; or pred_dir
    loop_pat = r"for\s+entry_idx.*in\s+range\(.*\):"
    loop_m = re.search(loop_pat, txt)
    if not loop_m:
        print("[WARN] main eval loop not found; inverse-transform not injected.")
        return txt, False

    # Try to find 'pred_price' assignment (or usage). If present, inject before it.
    anchor = re.search(r"\npred_price\s*=", txt[loop_m.start():])
    if not anchor:
        anchor = re.search(r"\npred_dir\s*=", txt[loop_m.start():])

    if anchor:
        abs_pos = loop_m.start() + anchor.start()
        new_txt = txt[:abs_pos] + INVERSE_SNIPPET + txt[abs_pos:]
        print("[OK] Inverse-transform block injected before pred_price/pred_dir.")
        return new_txt, True

    print("[WARN] Could not find 'pred_price' nor 'pred_dir' anchor; inverse-transform NOT injected.")
    print("      Please paste this snippet where you compute y_hat_norm (normalized delta):\n" + INVERSE_SNIPPET)
    return txt, False

def main():
    if not os.path.exists(TARGET):
        print(f"[ERR] {TARGET} not found in current directory.")
        sys.exit(1)

    src = load_text(TARGET)
    bak = backup(TARGET)
    print(f"[OK] Backup saved: {bak}")

    changed = False

    # 1) argparse flags
    txt, ok1 = inject_argparse_flags(src)
    changed = changed or ok1

    # 2) fetch_ohlcv CSV wrapper
    txt, ok2 = patch_fetch_ohlcv(txt)
    changed = changed or ok2

    # 3) inverse-transform
    txt, ok3 = try_inject_inverse(txt)
    changed = changed or ok3

    if changed:
        save_text(TARGET, txt)
        print(f"[OK] Patched {TARGET}.")
    else:
        print("[INFO] No changes were applied (file may already be patched).")


if __name__ == "__main__":
    main()
