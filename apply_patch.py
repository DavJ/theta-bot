
#!/usr/bin/env python3
import sys, re, pathlib

def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def write(p, s):
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)

def ensure_numpy_import(src):
    if re.search(r'^\s*import\s+numpy\s+as\s+np\b', src, flags=re.M):
        return src, False
    # vlož import numpy as np hned za první blok importů
    m = re.search(r'^(import .+|from .+ import .+)(?:\r?\n(import .+|from .+ import .+))*', src, flags=re.M)
    if m:
        end = m.end()
        return src[:end] + "\nimport numpy as np\n" + src[end:], True
    else:
        return "import numpy as np\n"+src, True

def add_args(src):
    changed = False
    # najdi řádek s --phase a vlož nové argumenty hned za něj, pokud nejsou
    if "--pred-ensemble" in src and "--max-by" in src:
        return src, changed
    pat = r'(parser\.add_argument\(\s*"--phase"[^)]+\)\s*)'
    ins = (
        'parser.add_argument("--pred-ensemble", choices=["avg","max"], default="avg",\\n'
        '                    help="How to combine per-ψ contributions: avg (sum over ψ) or max (take dominant ψ per step).")\\n'
        'parser.add_argument("--max-by", choices=["contrib","transform"], default="contrib",\\n'
        '                    help="For --pred-ensemble max: select dominant ψ by |X_j*β_j| (contrib) or |X_j| (transform).")\\n'
    )
    def repl(m):
        nonlocal changed
        changed = True
        return m.group(1) + "\\n" + ins
    src2, n = re.subn(pat, repl, src, count=1, flags=re.S)
    return (src2 if n else src), (changed or bool(n))

def ensure_ARGS(src):
    if re.search(r'^\s*ARGS\s*=\s*vars\(\s*args\s*\)', src, flags=re.M):
        return src, False
    # vlož global ARGS a ARGS = vars(args) hned za parse_args()
    pat = r'(args\s*=\s*parser\.parse_args\(\s*\)\s*)'
    ins = 'global ARGS\\nARGS = vars(args)\\n'
    src2, n = re.subn(pat, r'\\1\\n'+ins, src, count=1)
    return (src2 if n else src), bool(n)

def replace_pred(src):
    # pokus 1: najdi přesný řetězec "pred = X @ beta"
    if "pred = X @ beta" in src and "contrib =" in src:
        # pravděpodobně už je patch aplikován
        return src, False
    pat_exact = r'(\\npred\\s*=\\s*X\\s*@\\s*beta[^\\n]*\\n)'
    block = (
    "\n# contributions per ψ (T x nP): každý sloupec je příspěvek jedné periody\n"
    "contrib = X * beta.reshape(1, -1)\n\n"
    "if ARGS.get(\"pred_ensemble\", \"avg\") == \"avg\":\n"
    "    # původní chování (součet přes ψ)\n"
    "    pred = contrib.sum(axis=1)\n"
    "else:\n"
    "    # 'max' režim – vybíráme dominantní ψ pro každý časový krok\n"
    "    if ARGS.get(\"max_by\", \"contrib\") == \"transform\":\n"
    "        idx = np.abs(X).argmax(axis=1)\n"
    "    else:\n"
    "        idx = np.abs(contrib).argmax(axis=1)\n"
    "    rows = np.arange(contrib.shape[0])\n"
    "    pred = contrib[rows, idx]\n"
    )
    src2, n = re.subn(pat_exact, "\\n"+block, src, count=1)
    if n:
        return src2, True
    # pokus 2: obecně – najdi řádek s '@ beta' a nahrad
    pat_any = r'(\\npred\\s*=\\s*.*@\\s*beta[^\\n]*\\n)'
    src3, n2 = re.subn(pat_any, "\\n"+block, src, count=1)
    return (src3 if n2 else src), bool(n2)

def main():
    if len(sys.argv) < 2:
        print("Usage: python apply_patch.py path/to/theta_eval_hbatch_biquat.py")
        sys.exit(2)
    path = pathlib.Path(sys.argv[1])
    src = read(path)
    orig = src

    src, imp_changed = ensure_numpy_import(src)
    src, args_changed = add_args(src)
    src, ARGS_changed = ensure_ARGS(src)
    src, pred_changed = replace_pred(src)

    if src == orig:
        print("[info] Nic k úpravě (možná už je patch aplikován).")
    else:
        # záloha
        bak = path.with_suffix(path.suffix + ".bak")
        with open(bak, "w", encoding="utf-8") as f:
            f.write(orig)
        write(path, src)
        print("[ok] Uloženo. Záloha ->", bak.name)
        print(f"[detail] import numpy: {imp_changed}, argparse lines: {args_changed}, ARGS export: {ARGS_changed}, pred block: {pred_changed}")

if __name__ == "__main__":
    main()
