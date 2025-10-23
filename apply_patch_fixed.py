#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Přidá do theta_eval_hbatch_biquat.py nové CLI volby:
  --pred-ensemble {avg,max}   (default: avg)
  --max-by {transform,contrib} (default: transform)

Patch jen rozšíří argparse + nastaví global ARGS.
Chování modelu zůstává jako 'avg', takže příznaky zatím jen nevyhodí chybu.
"""

import sys, os, re

def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_patch_fixed.py <path/to/theta_eval_hbatch_biquat.py>")
        sys.exit(1)
    target = sys.argv[1]
    if not os.path.exists(target):
        print(f"[error] File not found: {target}")
        sys.exit(2)
    with open(target, "r", encoding="utf-8") as f:
        txt = f.read()

    changed = False

    # 1) global ARGS po parse_args()
    if "args = parser.parse_args()" in txt and "ARGS = vars(args)" not in txt:
        txt = txt.replace(
            "args = parser.parse_args()",
            "args = parser.parse_args()\n    global ARGS\n    ARGS = vars(args)"
        )
        changed = True

    # 2) dvě nové argparse volby, vložíme poblíž --phase
    phase_pat = re.compile(r'parser\\.add_argument\\(\\s*"--phase"[^)]*\\)', re.MULTILINE)
    m = phase_pat.search(txt)
    if m:
        insert_pos = m.end()
        if "--pred-ensemble" not in txt:
            addition = (
                '\n    parser.add_argument("--pred-ensemble", choices=["avg","max"], '
                'default="avg", help="How to combine per-ψ predictions: avg (default) or max")\n'
            )
            txt = txt[:insert_pos] + addition + txt[insert_pos:]
            changed = True
        if "--max-by" not in txt:
            anchor = txt.find("\n", insert_pos)
            if anchor == -1: anchor = insert_pos
            addition2 = (
                '    parser.add_argument("--max-by", choices=["transform","contrib"], '
                'default="transform", help="Criterion for max: |θ_k| (transform) or |β_k·θ_k| (contrib)")\n'
            )
            txt = txt[:anchor+1] + addition2 + txt[anchor+1:]
            changed = True
    else:
        # Fallback: vlož před --out
        out_pat = re.compile(r'parser\\.add_argument\\(\\s*"--out"[^)]*\\)', re.MULTILINE)
        m2 = out_pat.search(txt)
        if m2 and ("--pred-ensemble" not in txt or "--max-by" not in txt):
            ins = []
            if "--pred-ensemble" not in txt:
                ins.append('    parser.add_argument("--pred-ensemble", choices=["avg","max"], default="avg", '
                           'help="How to combine per-ψ predictions: avg (default) or max")')
            if "--max-by" not in txt:
                ins.append('    parser.add_argument("--max-by", choices=["transform","contrib"], default="transform", '
                           'help="Criterion for max: |θ_k| (transform) or |β_k·θ_k| (contrib)")')
            block = ("\n".join(ins) + "\n")
            txt = txt[:m2.start()] + block + txt[m2.start():]
            changed = True

    if not changed:
        print("[info] Nic k úpravě (vlajky už tam nejspíš jsou, nebo se nenašla kotva).")
    else:
        backup = target + ".bak"
        with open(backup, "w", encoding="utf-8") as fb:
            fb.write(txt)
        with open(target, "w", encoding="utf-8") as fw:
            fw.write(txt)
        print(f"[ok] Patched CLI v: {target}")
        print(f"[ok] Záloha: {backup}")
        print("[note] Zatím se stále používá průměr (avg). 'Max' logiku dopíšeme v další změně.")

if __name__ == "__main__":
    main()

