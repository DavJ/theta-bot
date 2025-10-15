#!/usr/bin/env bash
set -e

SYMBOL=BTCUSDT
INTERVAL=5m
LIMIT=20000
UPPER_GRID="0.504,0.508,0.512"
LOWER_GRID="0.496,0.492,0.488"
FEE=0.00036

python -m tests_backtest.cli.run_theta_benchmark   --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT"   --variants raw   --models logreg   --upper-grid "$UPPER_GRID" --lower-grid "$LOWER_GRID"   --fee-side "$FEE"   --outdir reports_cmp_raw

python -m tests_backtest.cli.run_theta_benchmark   --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT"   --variants theta_fft_fast   --models ckalman   --theta-K 48 --theta-tau 0.12   --upper-grid "$UPPER_GRID" --lower-grid "$LOWER_GRID"   --fee-side "$FEE"   --outdir reports_cmp_1D

python -m tests_backtest.cli.run_theta_benchmark   --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT"   --variants theta_fft_hybrid   --models ckalman   --theta-K 48 --theta-tau 0.12 --theta-tau-re 0.03   --upper-grid "$UPPER_GRID" --lower-grid "$LOWER_GRID"   --fee-side "$FEE"   --outdir reports_cmp_2D

python -m tests_backtest.cli.run_theta_benchmark   --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT"   --variants theta_fft_dynamic   --models ckalman   --theta-K 48 --theta-tau 0.12 --theta-tau-re 0.03   --theta-beta-re 0.02 --theta-beta-im 0.01   --upper-grid "$UPPER_GRID" --lower-grid "$LOWER_GRID"   --fee-side "$FEE"   --outdir reports_cmp_3D

python -m tests_backtest.cli.run_theta_benchmark   --symbol "$SYMBOL" --interval "$INTERVAL" --limit "$LIMIT"   --variants wavelet   --models logreg   --upper-grid "$UPPER_GRID" --lower-grid "$LOWER_GRID"   --fee-side "$FEE"   --outdir reports_cmp_wavelet

python - <<'PY'
import os, csv, glob

OUTS=[("raw","reports_cmp_raw"),("theta1D","reports_cmp_1D"),("theta2D","reports_cmp_2D"),("theta3D","reports_cmp_3D"),("wavelet","reports_cmp_wavelet")]

def parse_md_table(path):
    rows=[]; hdr=[]
    with open(path,'r',encoding='utf-8') as f:
        lines=[ln.strip() for ln in f]
    idx=None
    for i,l in enumerate(lines):
        if l.startswith("|") and "variant" in l:
            idx=i; break
    if idx is None: return hdr, rows
    hdr=[c.strip() for c in lines[idx].strip("|").split("|")]
    for l in lines[idx+2:]:
        if not l.startswith("|"): break
        rows.append([c.strip() for c in l.strip("|").split("|")])
    return hdr, rows

def best_from_md(path):
    hdr, rows = parse_md_table(path)
    if not rows: return None
    H={k:i for i,k in enumerate(hdr)}
    def get(r,k,typ=float):
        try: return typ(r[H[k]])
        except: return None
    scored=[]
    for r in rows:
        scored.append({
            "variant": r[H.get("variant",0)] if "variant" in H else "",
            "model":   r[H.get("model",1)] if "model" in H else "",
            "upper":   get(r,"upper"), "lower": get(r,"lower"),
            "sharpe":  get(r,"sharpe"), "maxdd": get(r,"maxdd"),
            "cagr":    get(r,"cagr"), "total_return": get(r,"total_return"),
            "trades":  int(get(r,"trades",float) or 0), "fees": get(r,"fees")
        })
    scored.sort(key=lambda d:((d["sharpe"] or -1e9),(d["total_return"] or -1e9),(d["trades"] or -1e9)), reverse=True)
    return scored[0]

def best_from_grids(outdir):
    best=None
    for path in glob.glob(os.path.join(outdir,"grid_*.csv")):
        with open(path,newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    cand={
                        "variant": os.path.basename(path).split("grid_")[1].split("_")[0],
                        "model":   os.path.basename(path).split("grid_")[1].split("_")[1],
                        "upper": float(row["upper"]), "lower": float(row["lower"]),
                        "sharpe": float(row["sharpe_mean"]), "maxdd": float(row["maxdd_mean"]),
                        "cagr": float(row["cagr_mean"]), "total_return": float(row["total_return_mean"]),
                        "trades": int(float(row["trades_sum"])), "fees": float(row["fees_sum"]),
                    }
                except: continue
                key=lambda d:(d["sharpe"], d["total_return"], d["trades"])
                if best is None or key(cand)>key(best): best=cand
    return best

def pick_best(outdir):
    for name in ("combined_best.md","summary.md"):
        p=os.path.join(outdir,name)
        if os.path.exists(p):
            b=best_from_md(p)
            if b: return b
    return best_from_grids(outdir)

os.makedirs("reports_cmp",exist_ok=True)
rows=[]
for label,outdir in OUTS:
    b = pick_best(outdir) or {"variant":"","model":"","upper":"","lower":"","sharpe":"","maxdd":"","cagr":"","total_return":"","trades":"","fees":""}
    b["label"]=label; rows.append(b)

with open("reports_cmp/overall.md","w",encoding="utf-8") as f:
    f.write("# Theta – Srovnání RAW vs 1D vs 2D vs 3D vs Wavelet\n\n")
    f.write("| label | variant | model | upper | lower | sharpe | maxdd | cagr | total_return | trades | fees |\n")
    f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for d in rows:
        f.write(f"| {d['label']} | {d['variant']} | {d['model']} | {d['upper']} | {d['lower']} | {d['sharpe']} | {d['maxdd']} | {d['cagr']} | {d['total_return']} | {d['trades']} | {d['fees']} |\n")
print("Hotovo -> reports_cmp/overall.md")
PY
