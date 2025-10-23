
import argparse, os, json
import numpy as np
import pandas as pd
from pathlib import Path

from tests_backtest.lib.theta_biquat_basis import forecast_theta_biquat_basis

def load_prices_from_csv(path):
    df = pd.read_csv(path)
    # Expect columns: time, open, high, low, close, volume (Binance-like)
    # Fallbacks:
    for c in ['close','c','Close','CLOSE']:
        if c in df.columns:
            close = df[c].astype(float).to_numpy()
            break
    else:
        raise ValueError("CSV must contain a 'close' (or 'c') column.")
    # time column optional
    ts = None
    for tcol in ['time','timestamp','open_time','t']:
        if tcol in df.columns:
            ts = df[tcol].to_numpy()
            break
    if ts is None:
        ts = np.arange(len(close))
    return ts, close

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="OHLCV CSV with a close column")
    ap.add_argument("--variants", default="raw,thetaBiquatBasis")
    ap.add_argument("--horizons", default="1h", help="Comma list, e.g. 1h,1d (we only use first here)")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--horizon-alpha", type=float, default=1.0)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--N-even", type=int, default=6)
    ap.add_argument("--N-odd", type=int, default=6)
    ap.add_argument("--omega", type=float, default=None)
    ap.add_argument("--ema-alpha", type=float, default=0.0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--outdir", default="reports_forecast")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ts, close = load_prices_from_csv(args.csv)

    # parse first horizon only for simplicity
    def parse_h(hs):
        h = hs.strip().lower()
        if h.endswith('h'):
            return int(h[:-1]) * (60//15) if (60%15)==0 else int(h[:-1])*4
        if h.endswith('d'):
            return int(h[:-1]) * 24 * (60//15)
        return int(h)
    H = parse_h(args.horizons.split(',')[0])

    preds, trues, meta = forecast_theta_biquat_basis(
        close, H=H, window=args.window, sigma=args.sigma,
        N_even=args.N_even, N_odd=args.N_odd, omega=args.omega,
        ema_alpha=args.ema_alpha, ridge=args.ridge
    )

    # Build long-form dataframe compatible with eval scripts (approx):
    # columns: variant, horizon_bars, y_true, y_pred
    rows = []
    start = args.window + H - 1
    for i, (yhat, y) in enumerate(zip(preds, trues)):
        # raw: naive hold last
        yraw = close[start+i - (H-1) - 1] if (start+i - (H-1) - 1) >= 0 else close[start+i-1]
        rows.append({"variant":"raw", "horizon_bars": H, "y_true": float(y), "y_pred": float(yraw)})
        rows.append({"variant":"thetaBiquatBasis", "horizon_bars": H, "y_true": float(y), "y_pred": float(yhat)})
    out = pd.DataFrame(rows)
    sym = Path(args.csv).stem
    out_csv = os.path.join(args.outdir, f"forecast_{sym}_win{args.window}_h{H}_biqbasis.csv")
    out.to_csv(out_csv, index=False)

    with open(os.path.join(args.outdir, f"forecast_{sym}_win{args.window}_h{H}_biqbasis.json"), "w") as f:
        json.dump({"meta": meta, "args": vars(args)}, f, indent=2)

    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
