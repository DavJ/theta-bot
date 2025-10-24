#!/usr/bin/env python3
import argparse, json, os, sys, math, numpy as np, pandas as pd
from dataclasses import dataclass

def ridge(X, y, lam):
    XT = X.T
    A = XT @ X
    n = A.shape[0]
    A.flat[::n+1] += lam
    b = XT @ y
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return beta

def periods_linspace(minP, maxP, nP):
    return np.linspace(minP, maxP, nP, dtype=float)

def build_basis(t_idx, periods):
    t = np.asarray(t_idx, dtype=float)
    cols = []
    for P in periods:
        w = 2.0 * math.pi / P
        cols.append(np.sin(w * t))
        cols.append(np.cos(w * t))
    return np.vstack(cols).T  # (T, 2*nP)

def gaussian_weights(n, sigma):
    if sigma <= 0:
        return np.ones(n)
    x = np.arange(n) - (n-1)
    w = np.exp(-(x**2)/(2.0*(sigma*(n/10.0))**2))
    return w / (w.sum() + 1e-12)

@dataclass
class EvalResult:
    hit_rate_pred: float
    hit_rate_hold: float
    corr_pred_true: float
    mae_price: float
    mae_return: float
    count: int

def metrics(last_prices, pred, true):
    pred_dir = np.sign(pred)
    true_dir = np.sign(true)
    hit_pred = (pred_dir == true_dir).mean() if len(true) else float('nan')
    hold_ret = (true / (last_prices + 1e-12))
    hold_up = (hold_ret > 0).astype(int)
    hit_hold = hold_up.mean() if len(hold_up) else float('nan')
    c = np.corrcoef(pred, true)[0,1] if len(true) > 1 else float('nan')
    mae_p = np.mean(np.abs(true)) if len(true) else float('nan')
    mae_r = np.mean(np.abs(true/(last_prices+1e-12))) if len(true) else float('nan')
    return EvalResult(hit_pred, hit_hold, c, mae_p, mae_r, len(true))

def fetch_csv(path, time_col, close_col):
    df = pd.read_csv(path)
    if time_col not in df.columns or close_col not in df.columns:
        raise ValueError(f"CSV '{path}' must contain columns '{time_col}' and '{close_col}'. Columns: {list(df.columns)}")
    out = df[[time_col, close_col]].rename(columns={time_col:'time', close_col:'close'}).copy()
    out['time'] = pd.to_datetime(out['time'], utc=True, errors='coerce')
    out = out.dropna().sort_values('time').reset_index(drop=True)
    return out

def is_csv_symbol(s):
    return s.lower().endswith('.csv')

def evaluate_symbol_csv(path, window, horizon, minP, maxP, nP, sigma, lam, pred_ensemble, max_by):
    df = fetch_csv(path, ARGS.csv_time_col or 'time', ARGS.csv_close_col or 'close')

    closes = df['close'].values.astype(float)
    times  = df['time'].values
    periods = periods_linspace(minP, maxP, nP)
    t_idx = np.arange(len(closes), dtype=float)
    X_all = build_basis(t_idx, periods)

    rows, preds, trues, lasts = [], [], [], []

    for compare_idx in range(window, len(closes)-horizon):
        entry_idx = compare_idx - 1
        last_price = closes[entry_idx]
        future_price = closes[compare_idx + horizon - 1]
        true_delta = future_price - last_price

        lo = compare_idx - window
        hi = compare_idx
        Xw = X_all[lo:hi, :]
        yw = (closes[lo+horizon:hi+horizon] - closes[lo:hi]).astype(float)
        m = min(len(Xw), len(yw))
        Xw = Xw[:m, :]
        yw = yw[:m]

        w = gaussian_weights(m, sigma)
        Xw_w = Xw * w[:,None]
        yw_w = yw * w

        beta = ridge(Xw_w, yw_w, lam)
        x_now = X_all[entry_idx, :]

        # dominance per period (two features sin/cos):
        contrib_per_P = []
        for iP in range(len(periods)):
            s = x_now[2*iP+0] * beta[2*iP+0]
            c = x_now[2*iP+1] * beta[2*iP+1]
            if max_by == 'transform':
                contrib_per_P.append(abs(s) + abs(c))
            else:
                contrib_per_P.append(abs(s + c))

        if pred_ensemble == 'avg':
            pred_delta = float(x_now @ beta)
        else:
            k = int(np.argmax(contrib_per_P))
            pred_delta = float(x_now[2*k:2*k+2] @ beta[2*k:2*k+2])

        
# === added (no look-ahead): extra analytics columns ===
pred_dir = int(np.sign(pred_delta))
true_dir = int(np.sign(true_delta))
correct_pred_val = 1 if (pred_dir != 0 and pred_dir == true_dir) else 0
hold_ret = (future_price - last_price) / last_price if last_price != 0.0 else 0.0
# === end added ===
rows.append({
            'time': str(times[entry_idx]),
            'entry_idx': int(entry_idx),
            'compare_idx': int(compare_idx),
            'last_price': float(last_price),
            'pred_price': float(last_price + pred_delta),
            'future_price': float(future_price),
            'pred_delta': float(pred_delta),
            'true_delta': float(true_delta),
            'pred_dir': int(pred_dir),
            'true_dir': int(true_dir),
            'correct_pred': int(correct_pred_val),
            'hold_ret': float(hold_ret),
        })
        preds.append(pred_delta); trues.append(true_delta); lasts.append(last_price)

    if not rows:
        raise RuntimeError("Not enough rows after window/horizon to evaluate.")

    out_df = pd.DataFrame(rows)
    res = metrics(np.asarray(lasts), np.asarray(preds), np.asarray(trues))
    return out_df, res

def run_batch(symbols_str, **kw):
    syms = [s.strip() for s in symbols_str.split(',') if s.strip()]
    records = []
    for sym in syms:
        print(f"\n=== Running {sym} ===\n")
        if not is_csv_symbol(sym):
            print(f"[warn] Only CSV paths are supported in this standalone evaluator. Skipping '{sym}'.")
            continue
        df, res = evaluate_symbol_csv(sym, **kw)
        base = os.path.basename(sym).replace('.','').replace('/','')
        eval_csv = f"eval_h_{base}.csv"
        sum_json = f"sum_h_{base}.json"
        df.to_csv(eval_csv, index=False)
        with open(sum_json, 'w') as f:
            json.dump({
                'hit_rate_pred': res.hit_rate_pred,
                'hit_rate_hold': res.hit_rate_hold,
                'corr_pred_true': res.corr_pred_true,
                'mae_price': res.mae_price,
                'mae_return': res.mae_return,
                'count': res.count
            }, f, indent=2)
        print("\n--- HSTRATEGY vs HOLD ---")
        print(f"hit_rate_pred:  {res.hit_rate_pred:.6f}")
        print(f"hit_rate_hold:  {res.hit_rate_hold:.6f}")
        print(f"corr_pred_true: {res.corr_pred_true:.6f}")
        print(f"mae_price:      {res.mae_price}")
        print(f"mae_return:     {res.mae_return}")
        print(f"count:          {res.count}\n")
        print(f"Uloženo CSV: {eval_csv}")
        print(f"Uloženo summary: {sum_json}\n")

        records.append({
            'symbol': sym,
            'phase': 'biquat' if ARGS.phase in ('biquat','complex') else ARGS.phase,
            'pred_ensemble': ARGS.pred_ensemble,
            'max_by': ARGS.max_by if ARGS.pred_ensemble=='max' else '',
            'hit_rate_pred': res.hit_rate_pred,
            'hit_rate_hold': res.hit_rate_hold,
            'delta_hit': res.hit_rate_pred - res.hit_rate_hold,
            'corr_pred_true': res.corr_pred_true,
            'mae_price': res.mae_price,
            'mae_return': res.mae_return,
            'count': res.count
        })
    return pd.DataFrame.from_records(records)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', required=True, help='Comma-separated CSV paths')
    p.add_argument('--csv-time-col', default=None, dest='csv_time_col', help='Time column name (default: time)')
    p.add_argument('--csv-close-col', default=None, dest='csv_close_col', help='Close column name (default: close)')
    p.add_argument('--interval', default='1h')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--minP', type=int, default=24)
    p.add_argument('--maxP', type=int, default=480)
    p.add_argument('--nP', type=int, default=12)
    p.add_argument('--sigma', type=float, default=0.8)
    p.add_argument('--lambda', dest='lam', type=float, default=1e-3)
    p.add_argument('--limit', type=int, default=2000)
    p.add_argument('--phase', choices=['simple','complex','biquat'], default='biquat')
    p.add_argument('--pred-ensemble', choices=['avg','max'], default='avg', help='avg (all Ps) or max (dominant P)')
    p.add_argument('--max-by', choices=['transform','contrib'], default='transform', help='criterion for --pred-ensemble max')
    p.add_argument('--out', required=True, help='Summary CSV path')
    return p.parse_args()

def main():
    global ARGS
    ARGS = parse_args()
    df = run_batch(
        ARGS.symbols,
        window=ARGS.window,
        horizon=ARGS.horizon,
        minP=ARGS.minP,
        maxP=ARGS.maxP,
        nP=ARGS.nP,
        sigma=ARGS.sigma,
        lam=ARGS.lam,
        pred_ensemble=ARGS.pred_ensemble,
        max_by=ARGS.max_by,
    )
    if len(df):
        df.to_csv(ARGS.out, index=False)
        print(f"\nUloženo: {ARGS.out}")
        print(df.to_string(index=False))
    else:
        print("[warn] No records written to summary (did you pass CSV paths?)")

if __name__ == '__main__':
    main()
