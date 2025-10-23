#!/usr/bin/env python3
import sys, re, shutil, os

def ensure_flags(src: str) -> str:
    if "--pred-ensemble" in src and "--max-by" in src:
        return src
    pattern = r'(parser\.add_argument\([^\n]*"--phase"[^\n]*\)\s*)'
    flags = (
        '    parser.add_argument("--pred-ensemble", choices=["avg","max"], default="avg", '
        'help="How to combine per-ψ predictions: avg (default) or max")\n'
        '    parser.add_argument("--max-by", choices=["transform","contrib"], default="transform", '
        'help="Criterion for max: |θ_k| (transform) or |β_k·θ_k| (contrib)")\n'
    )
    return re.sub(pattern, r"\1\n" + flags, src)

def ensure_ARGS(src: str) -> str:
    if "ARGS = vars(args)" in src:
        return src
    return src.replace(
        "args = parser.parse_args()",
        "args = parser.parse_args()\n    global ARGS\n    ARGS = vars(args)"
    )

def patch_pred_logic(src: str) -> str:
    block = '''
# === biquat-max patch begin ===
import numpy as _np
try:
    _pred_ens = ARGS.get("pred_ensemble", "avg")
    _max_by = ARGS.get("max_by", "transform")
except Exception:
    _pred_ens, _max_by = "avg", "transform"

if _pred_ens == "max":
    try:
        if _max_by == "contrib":
            _idx = int(_np.argmax(_np.abs(beta * theta_vals)))
        else:
            _idx = int(_np.argmax(_np.abs(theta_vals)))
        pred_delta = comp[_idx]
    except Exception:
        pred_delta = _np.mean(comp)
else:
    pred_delta = _np.mean(comp)
# === biquat-max patch end ===
'''
    pat = re.compile(r'\n[ \t]*pred_delta\s*=\s*_?np\.mean\(\s*comp\s*\)\s*\n')
    m = pat.search(src)
    if not m:
        return src + "\n\n" + block
    return src[:m.start()] + block + src[m.end():]

def main():
    if len(sys.argv) < 2:
        print("Usage: apply_patch_fixed.py <path/to/theta_eval_hbatch_biquat.py>")
        sys.exit(1)
    path = sys.argv[1]
    code = open(path, encoding="utf-8").read()
    shutil.copyfile(path, path + ".bak")
    code = ensure_flags(code)
    code = ensure_ARGS(code)
    code = patch_pred_logic(code)
    open(path, "w", encoding="utf-8").write(code)
    print("[ok] Patch applied successfully. Backup created:", path + ".bak)

if __name__ == "__main__":
    main()

