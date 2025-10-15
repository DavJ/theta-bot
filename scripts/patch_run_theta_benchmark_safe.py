#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe autopatcher for tests_backtest/cli/run_theta_benchmark.py
- Adds import of transforms_theta_fast
- Adds argparse args: --theta-tau-re, --theta-beta-re, --theta-beta-im
- Inserts new 'elif' branches for: theta_fft_fast, theta_fft_hybrid, theta_fft_dynamic
The script is idempotent and creates a .bak backup.
"""
import io, os, re, sys

CLI = "tests_backtest/cli/run_theta_benchmark.py"

IMPORT_BLOCK = """from tests_backtest.common.transforms_theta_fast import (
    theta_fft_fast, theta_fft_hybrid, theta_fft_dynamic
)
"""

ARG_BLOCK = """ap.add_argument('--theta-tau-re', type=float, default=0.03)
ap.add_argument('--theta-beta-re', type=float, default=0.0)
ap.add_argument('--theta-beta-im', type=float, default=0.0)
"""

VARIANT_BLOCK = """elif variant=='theta_fft_fast':
        z, _ = theta_fft_fast(seg, K=theta_K, tau_im=theta_tau)
        feats_c.append(z); f = np.zeros(1)

    elif variant=='theta_fft_hybrid':
        z, _ = theta_fft_hybrid(seg, K=theta_K, tau_re=args.theta_tau_re, tau_im=theta_tau)
        feats_c.append(z); f = np.zeros(1)

    elif variant=='theta_fft_dynamic':
        z, _ = theta_fft_dynamic(
            seg, K=theta_K,
            tau_re0=args.theta_tau_re, tau_im0=theta_tau,
            beta_re=args.theta_beta_re, beta_im=args.theta_beta_im
        )
        feats_c.append(z); f = np.zeros(1)
"""

def read(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()

def write(path, data):
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(data)

def ensure_imports(src):
    if "transforms_theta_fast" in src:
        return src, False
    # Try to insert after any existing transforms_* import or after numpy import
    pat1 = re.compile(r"(^from\s+tests_backtest\.common\.transforms_[^\n]+\n)", re.M)
    m = pat1.search(src)
    if m:
        pos = m.end()
        return src[:pos] + IMPORT_BLOCK + src[pos:], True
    # fallback: after "import numpy as np"
    pat2 = re.compile(r"(^import\s+numpy\s+as\s+np\s*\n)", re.M)
    m = pat2.search(src)
    if m:
        pos = m.end()
        return src[:pos] + IMPORT_BLOCK + src[pos:], True
    # fallback: at top after first import line
    pat3 = re.compile(r"(^import\s+[^\n]+\n)", re.M)
    m = pat3.search(src)
    if m:
        pos = m.end()
        return src[:pos] + IMPORT_BLOCK + src[pos:], True
    # last resort: prepend
    return IMPORT_BLOCK + src, True

def ensure_args(src):
    if "--theta-tau-re" in src:
        return src, False
    # find the line that defines --theta-tau and insert after it
    pat = re.compile(r"(ap\.add_argument\('--theta\-tau'[^)]*\)\s*\n)", re.M)
    m = pat.search(src)
    if not m:
        # try to find any argparse block near other theta args
        pat2 = re.compile(r"(ap\.add_argument\('[^']*theta[^']*'[^)]*\)\s*\n)", re.M)
        m = pat2.search(src)
    if m:
        pos = m.end()
        return src[:pos] + ARG_BLOCK + src[pos:], True
    # fallback: append to end (suboptimal but works)
    return src + "\n" + ARG_BLOCK + "\n", True

def ensure_variants(src):
    if "variant=='theta_fft_hybrid'" in src:
        return src, False
    # Prefer to insert after fft_complex branch
    pat = re.compile(r"(elif\s+variant==['\"]fft_complex['\"]\s*:[^\n]*\n)", re.M)
    m = pat.search(src)
    if m:
        pos = m.end()
        return src[:pos] + VARIANT_BLOCK + src[pos:], True
    # Try to find any other fft_* branch
    pat2 = re.compile(r"(elif\s+variant==['\"]fft_[^'\"]+['\"]\s*:[^\n]*\n)", re.M)
    m = pat2.search(src)
    if m:
        pos = m.end()
        return src[:pos] + VARIANT_BLOCK + src[pos:], True
    # As a last resort, try to inject before 'return X, Xc, y, metas' inside build_dataset
    pat3 = re.compile(r"(return\s+X,\s*Xc,\s*y,\s*metas)", re.M)
    m = pat3.search(src)
    if m:
        pos = m.start(1)
        return src[:pos] + VARIANT_BLOCK + src[pos:], True
    # Nothing matched; append at end (user may have to adjust indent manually)
    return src + "\n# VARIANT BLOCK (adjust indent if needed)\n" + VARIANT_BLOCK + "\n", True

def main():
    if not os.path.exists(CLI):
        print(f"Soubor {CLI} nebyl nalezen. Spusť mě z kořene repozitáře.")
        sys.exit(1)
    src = read(CLI)
    bak = CLI + ".bak"
    with open(bak, "w", encoding="utf-8") as f:
        f.write(src)

    changed = False
    src, imp_changed = ensure_imports(src); changed = changed or imp_changed
    src, arg_changed = ensure_args(src);     changed = changed or arg_changed
    src, var_changed = ensure_variants(src); changed = changed or var_changed

    if changed:
        write(CLI, src)
        print("[OK] run_theta_benchmark.py upraven.")
        if imp_changed: print("  + přidány importy transforms_theta_fast")
        if arg_changed: print("  + přidány argparse parametry (--theta-tau-re, --theta-beta-re, --theta-beta-im)")
        if var_changed: print("  + přidány větve pro theta_fft_fast/hybrid/dynamic")
        print(f"[Backup] Původní soubor uložen jako: {bak}")
    else:
        print("[INFO] Nebyly potřeba změny (už je to zřejmě upravené).")

if __name__ == "__main__":
    main()
