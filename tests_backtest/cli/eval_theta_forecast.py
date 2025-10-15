
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import re

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multi-horizon forecast CSV by SSE/MSE (pure)")
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", type=str, default="reports_forecast")
    p.add_argument("--metric", choices=["SSE","MSE"], default="MSE")
    p.add_argument("--decay-half-life", dest="decay_half_life", type=float, default=None,
                   help="v počtu predikcí; pokud zadáš, použijí se exp. váhy (recentnější > starší)")
    return p.parse_args()

def exp_weights(n: int, half_life: float) -> np.ndarray:
    """index 0 = nejstarší, n-1 = nejnovější.
    váha(i) = 0.5 ** ((n-1 - i)/half_life)
    """
    i = np.arange(n, dtype=float)
    return 0.5 ** ((n-1 - i) / half_life)

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if "close" not in df.columns:
        raise SystemExit("CSV musí obsahovat 'close'.")

    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    rx = re.compile(r"^pred_(?P<variant>[a-zA-Z0-9]+)_h(?P<h>\d+)$")
    meta = []
    for c in pred_cols:
        m = rx.match(c)
        if not m:
            continue
        meta.append((c, m.group("variant"), int(m.group("h"))))
    if not meta:
        raise SystemExit("Nenašel jsem žádné sloupce 'pred_<variant>_h<hbars>'.")

    records = []
    for col, variant, H in meta:
        y_true_all = df["close"].shift(-H).to_numpy(dtype=float)
        y_pred_all = df[col].to_numpy(dtype=float)
        mask = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        y_true = y_true_all[idx]
        y_pred = y_pred_all[idx]
        err = y_pred - y_true
        if args.decay_half_life is not None and args.decay_half_life > 0:
            w = exp_weights(len(err), args.decay_half_life)
            sse = float(np.sum(w * (err**2)))
            eff_n = float(np.sum(w))
            mse = sse / eff_n if eff_n > 0 else float("nan")
        else:
            sse = float(np.sum(err**2))
            mse = float(np.mean(err**2))
        val = sse if args.metric == "SSE" else mse
        records.append({
            "variant": variant,
            "horizon_bars": H,
            "N": int(len(err)),
            "SSE": sse,
            "MSE": mse,
            "score": val
        })

    dfm = pd.DataFrame(records)
    if dfm.empty:
        raise SystemExit("Žádné validní metriky k vypsání.")
    dfm = dfm.sort_values(["score","MSE","SSE","variant","horizon_bars"])
    out_path = outdir / (Path(args.csv).stem + "_metrics_pure.csv")
    dfm.to_csv(out_path, index=False)

    print("\n=== Leaderboard (nižší lepší; default MSE) ===")
    print(dfm.to_string(index=False))
    print(f"\nUloženo: {out_path}")

if __name__ == "__main__":
    main()
