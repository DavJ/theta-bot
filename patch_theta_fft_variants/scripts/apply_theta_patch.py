#!/usr/bin/env python3
import io, os, re, sys
from pathlib import Path

TARGET = Path("tests_backtest/cli/run_theta_benchmark.py")

if not TARGET.exists():
    print(f"[ERROR] {TARGET} not found. Run from repo root.")
    sys.exit(1)

src = TARGET.read_text(encoding="utf-8")
bak = TARGET.with_suffix(TARGET.suffix + ".bak_full")
bak.write_text(src, encoding="utf-8")

changed = False

def ensure_imports(text: str):
    changed_local = False
    if "from tests_backtest.common.transforms_theta_fast import (theta_fft_fast, theta_fft_hybrid, theta_fft_dynamic)" not in text:
        text = re.sub(
            r"(from tests_backtest\.common\.transforms.*\n)",
            r"\1from tests_backtest.common.transforms_theta_fast import (theta_fft_fast, theta_fft_hybrid, theta_fft_dynamic)\n",
            text,
            count=1
        )
        changed_local = True
    return text, changed_local

def fix_build_dataset(text: str):
    changed_local = False
    text2 = re.sub(
        r"elif variant=='fft_complex':\s+elif variant=='theta_fft_fast':[^\n]*\n",
        "elif variant=='fft_complex':\n",
        text
    )
    if text2 != text:
        text = text2
        changed_local = True

    def ensure_branch(name, body):
        nonlocal text, changed_local
        if re.search(rf"elif\s+variant==['\"]{name}['\"]\s*:", text) is None:
            ins_pat = r"elif variant=='theta_complex'\s*:"
            m = re.search(ins_pat, text)
            snippet = "\n" + f"        elif variant=='{name}':\n" + body + "\n"
            if m:
                start = m.start()
                text = text[:start] + snippet + text[start:]
            else:
                text = re.sub(r"\n\s*else:\s*\n\s*raise ValueError\('unknown variant'\)\s*\n", snippet + r"\n        else:\n            raise ValueError('unknown variant')\n", text)
            changed_local = True

    body_fast = "            z, _ = theta_fft_fast(seg, K=theta_K, tau_im=theta_tau)\n            feats_c.append(z); f = np.zeros(1)"
    body_hybrid = "            z, _ = theta_fft_hybrid(seg, K=theta_K, tau_re=theta_tau_re, tau_im=theta_tau)\n            feats_c.append(z); f = np.zeros(1)"
    body_dynamic = "            z, _ = theta_fft_dynamic(seg, K=theta_K, tau_re0=theta_tau_re, tau_im0=theta_tau, beta_re=theta_beta_re, beta_im=theta_beta_im)\n            feats_c.append(z); f = np.zeros(1)"

    ensure_branch("theta_fft_fast", body_fast)
    ensure_branch("theta_fft_hybrid", body_hybrid)
    ensure_branch("theta_fft_dynamic", body_dynamic)

    return text, changed_local

def ensure_args(text: str):
    changed_local = False
    additions = [
        ("--theta-K", "ap.add_argument('--theta-K', type=int, default=16)"),
        ("--theta-tau", "ap.add_argument('--theta-tau', type=float, default=0.25)"),
        ("--theta-tau-re", "ap.add_argument('--theta-tau-re', type=float, default=0.03)"),
        ("--theta-beta-re", "ap.add_argument('--theta-beta-re', type=float, default=0.0)"),
        ("--theta-beta-im", "ap.add_argument('--theta-beta-im', type=float, default=0.0)"),
        ("--theta-ridge", "ap.add_argument('--theta-ridge', type=float, default=1e-3)"),
        ("--theta-gs", "ap.add_argument('--theta-gs', action='store_true')"),
    ]

    parse_pos = text.find("args = ap.parse_args()")
    if parse_pos == -1:
        # very defensive: append small block
        insertion = "\n" + "\n".join([a[1] for a in additions]) + "\n"
        text = text + insertion
        return text, True

    # try to insert before parse_args if missing
    for key, line in additions:
        if key not in text:
            text = text[:parse_pos] + line + "\n" + text[parse_pos:]
            parse_pos += len(line) + 1
            changed_local = True

    return text, changed_local

def ensure_allowlist(text: str):
    changed_local = False
    if "ALLOWED_COMPLEX" not in text:
        text = re.sub(
            r"(def build_dataset\(.*?\)\s*:[\s\S]*?return X, y, meta, None\s*\n)",
            r"\1\nALLOWED_COMPLEX = (\n"
            r"    'fft_complex',\n"
            r"    'theta_complex',\n"
            r"    'theta_fft_fast',\n"
            r"    'theta_fft_hybrid',\n"
            r"    'theta_fft_dynamic',\n"
            r")\n",
            text,
            count=1
        )
        changed_local = True

    # Inject filter "if model=='ckalman' and variant not in ALLOWED_COMPLEX: continue"
    pat = r"for variant in args\.variants:\s*\n\s*for model in args\.models:\s*\n"
    m = re.search(pat, text)
    if m and "ALLOWED_COMPLEX" in text and "variant not in ALLOWED_COMPLEX" not in text:
        insert_at = m.end()
        inject = "        if model == 'ckalman' and variant not in ALLOWED_COMPLEX:\n            continue\n"
        text = text[:insert_at] + inject + text[insert_at:]
        changed_local = True

    return text, changed_local

def add_debug_after_dataset(text: str):
    changed_local = False
    m = re.search(r"X,\s*y,\s*meta,\s*Xc_all\s*=\s*build_dataset\([^\)]*\)\s*", text)
    if m and "[debug] variant=" not in text:
        idx = m.end()
        debug = "\n    print(f\"[debug] variant={variant} model={model} | X={X.shape} y={y.shape} Xc={'None' if Xc_all is None else Xc_all.shape}\")\n"
        text = text[:idx] + debug + text[idx:]
        changed_local = True
    return text, changed_local

src, ch = ensure_imports(src); changed |= ch
src, ch = fix_build_dataset(src); changed |= ch
src, ch = ensure_args(src); changed |= ch
src, ch = ensure_allowlist(src); changed |= ch
src, ch = add_debug_after_dataset(src); changed |= ch

if changed:
    TARGET.write_text(src, encoding="utf-8")
    print("[OK] Patch applied to", TARGET)
    print("[INFO] Backup at", bak)
else:
    print("[INFO] No changes were necessary (file already patched).")
