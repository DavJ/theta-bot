#!/usr/bin/env python3
import itertools, subprocess, os, csv, json
from pathlib import Path

SYMS = "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT,XRPUSDT"
COMMON = [
    "--symbols", SYMS,
    "--interval", "1h",
    "--window", "256",
    "--horizon", "4",
    "--minP", "24",
    "--maxP", "480",
    "--limit", "2000",
]
pred_ensembles = ["avg"]
max_bys = ["transform"]
lambdas = ["1e-3","2e-3","3e-3"]
nps = ["8","12","16"]
dense_nps = ["", "16"]  # "" = bez dense běhu

out_dir = Path("sweep_out")
out_dir.mkdir(exist_ok=True)
summary_path = out_dir/"sweep_summary.csv"

with open(summary_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pred_ensemble","max_by","lambda","nP","dense_np",
                "corr_mean","hit_mean","corr_dense_mean","delta_corr_dense"])
    for pe, mb, lam, np_, dnp in itertools.product(pred_ensembles, max_bys, lambdas, nps, dense_nps):
        out = out_dir / f"robust_pe{pe}_mb{mb}_lam{lam}_np{np_}_d{dnp or 'none'}.csv"
        cmd = ["python", "robustness_suite_v2.py",
               *COMMON,
               "--sigma", "0.8",
               "--pred-ensemble", pe,
               "--max-by", mb,
               "--nP", np_,
               "--lambda", lam,
               "--out", str(out)]
        if dnp:
            cmd += ["--dense-np", dnp]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # načti csv a spočíte

