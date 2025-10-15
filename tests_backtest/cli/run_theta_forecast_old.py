import os, argparse, numpy as np, pandas as pd

def _download_to_csv(symbol, interval, csv_path, limit=10000, start=None, end=None):
    try:
        from tests_backtest.data.binance_download import download_to_csv
        download_to_csv(symbol, interval, csv_path, limit=limit, start_ms=start, end_ms=end)
        return True
    except Exception:
        return False

from tests_backtest.common.theta_transforms_min import raw_features, fft_reim, theta1D, theta2D, theta3D

def make_features(seg, variant, fft_topn, K, tau, tau_re, psi):
    if variant == "raw":
        return raw_features(seg)
    elif variant == "fft":
        return fft_reim(seg, topn=fft_topn)
    elif variant == "theta1D":
        return theta1D(seg, K=K, tau=tau)
    elif variant == "theta2D":
        return theta2D(seg, K=K, tau=tau, tau_re=tau_re)
    elif variant == "theta3D":
        return theta3D(seg, K=K, tau=tau, tau_re=tau_re, psi=psi)
    else:
        raise ValueError("Unknown variant: " + variant)

def ridge_fit_predict(Xtr, ytr, Xte, ridge):
    Xtr = np.asarray(Xtr, float); ytr = np.asarray(ytr, float)
    Xte = np.asarray(Xte, float)
    m = Xtr.shape[1]
    A = Xtr.T @ Xtr + ridge*np.eye(m)
    b = Xtr.T @ ytr
    w = np.linalg.solve(A, b)
    yhat = Xte @ w
    return yhat, w

def build_roll_forecast(df, variants, window, horizon, horizon_alpha, fft_topn, K, tau, tau_re, psi, ridge):
    closes = df['c'].astype(float).to_numpy()
    times = df['t'].astype(int).to_numpy()
    out_rows = []
    debug_files = {}

    for variant in variants:
        feats = []
        targets = []
        idxs = []
        for i in range(window, len(closes)-horizon):
            seg = closes[i-window:i]
            y_future = closes[i:i+horizon]
            w = np.array([horizon_alpha**h for h in range(horizon)], float)
            y_target = float((w @ y_future) / (w.sum()+1e-12))

            f = make_features(seg, variant, fft_topn, K, tau, tau_re, psi)
            feats.append(f); targets.append(y_target); idxs.append(i)

        feats = np.asarray(feats, float)
        targets = np.asarray(targets, float)

        preds, y_dbg, t_dbg = [], [], []
        start_j = max(32, window//4)
        for j in range(start_j, len(targets)):
            Xtr = feats[:j]; ytr = targets[:j]
            Xte = feats[j:j+1]
            yhat, _ = ridge_fit_predict(Xtr, ytr, Xte, ridge)
            preds.append(float(yhat[0])); y_dbg.append(float(targets[j])); t_dbg.append(int(times[window + j]))

        y_pred_seq = np.array(preds, float)
        y_true_seq = np.array(y_dbg, float)

        wmse = float(((y_true_seq - y_pred_seq)**2).mean())
        mae_v = float(np.abs(y_true_seq - y_pred_seq).mean())

        import pandas as pd
        dbg = pd.DataFrame({"t": t_dbg, "y_true": y_true_seq, "y_pred": y_pred_seq})
        debug_files[variant] = dbg

        out_rows.append({
            "variant": variant,
            "window": window,
            "horizon": horizon,
            "horizon_alpha": horizon_alpha,
            "ridge": ridge,
            "features": feats.shape[1] if feats.ndim==2 else 0,
            "n_samples": len(y_pred_seq),
            "mse_naive": float(np.mean((y_true_seq - np.mean(y_true_seq))**2)),
            "wmse": wmse,
            "mae": mae_v,
        })
    return out_rows, debug_files

def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)

def load_or_download(symbol, interval, limit, csv_path, outdir, start=None, end=None):
    if csv_path and os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
    else:
        ensure_outdir(outdir)
        path = os.path.join(outdir, f"forecast_{symbol}_{interval}.csv")
        ok = _download_to_csv(symbol, interval, path, limit=limit, start=start, end=end)
        if not ok:
            raise RuntimeError("Downloader not available. Provide --csv with columns t,o,h,l,c,v.")
        import pandas as pd
        df = pd.read_csv(path)
    keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(k in df.columns for k in keep):
        raise RuntimeError("CSV must contain columns: " + ",".join(keep))
    return df[keep]

def write_summary(outdir, rows):
    ensure_outdir(outdir)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "forecast_metrics.csv"), index=False)
    cols = ["variant","features","window","horizon","horizon_alpha","ridge","n_samples","wmse","mae","mse_naive"]
    lines = []
    lines.append("# Forecast Metrics Summary\n")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"]*len(cols)) + "|")
    for _,r in df.sort_values("wmse").iterrows():
        def fmt(c):
            if c in ("wmse","mae","mse_naive","horizon_alpha","ridge"):
                return f"{r[c]:.6f}"
            else:
                return f"{r[c]}"
        lines.append("| " + " | ".join([fmt(c) for c in cols]) + " |")
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--variants", nargs="+", default=["raw","fft","theta1D","theta2D","theta3D"])
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--horizon-alpha", type=float, default=0.9)
    ap.add_argument("--fft-topn", type=int, default=24)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.12)
    ap.add_argument("--tau-re", type=float, default=0.02)
    ap.add_argument("--psi", type=float, default=0.0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    df = load_or_download(args.symbol, args.interval, args.limit, args.csv, args.outdir)
    rows, debug = build_roll_forecast(df, args.variants, args.window, args.horizon, args.horizon_alpha,
                                      args.fft_topn, args.K, args.tau, args.tau_re, args.psi, args.ridge)
    write_summary(args.outdir, rows)
    for v,dfd in debug.items():
        dfd.to_csv(os.path.join(args.outdir, f"debug_{v}.csv"), index=False)
    print("Hotovo ->", args.outdir)

if __name__ == "__main__":
    main()
