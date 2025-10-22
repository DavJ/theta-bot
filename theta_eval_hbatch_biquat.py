
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_hbatch_biquat.py
---------------------------------
Batch evaluator with optional biquaternion phase basis.

USAGE (examples):
  python theta_eval_hbatch_biquat.py \
    --symbols BTCUSDT,ETHUSDT,BNBUSDT \
    --interval 1h --window 256 --horizon 4 \
    --minP 24 --maxP 480 --nP 12 \
    --sigma 0.8 --lambda 1e-3 --limit 2000 \
    --phase biquat \
    --out hbatch_summary.csv

Notes:
- If you pass a local CSV path instead of an exchange symbol (endswith .csv),
  the script will read columns: time (ISO or epoch ms) and close.
- Otherwise, it will try to fetch from Binance via `ccxt` if installed.
- Outputs per-symbol CSV (eval_h_<SYM>.csv) and JSON summary (sum_h_<SYM>.json),
  and the combined summary CSV (--out).
"""

import argparse
import math
import sys
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional ccxt import
_CCXT = None
try:
    import ccxt  # type: ignore
    _CCXT = ccxt
except Exception:
    _CCXT = None


# ---------- utilities ----------

def ema1(x: np.ndarray, alpha: float) -> np.ndarray:
    """Simple EMA with coefficient alpha in (0,1]; alpha=1 => no smoothing."""
    if alpha <= 0:  # no smoothing -> return original
        return x.copy()
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    a = float(alpha)
    b = 1.0 - a
    for i in range(1, len(x)):
        out[i] = a * x[i] + b * out[i-1]
    return out


def make_period_grid(minP: int, maxP: int, nP: int) -> np.ndarray:
    """Log-spaced periods between minP and maxP (inclusive-ish)."""
    return np.unique(
        np.round(np.exp(np.linspace(np.log(minP), np.log(maxP), nP))).astype(int)
    )


def build_fourier_basis(t: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Real sin/cos Fourier basis for given integer periods on grid t (0..W-1)."""
    W = len(t)
    cols = []
    for P in periods:
        w = 2 * np.pi / float(P)
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    X = np.stack(cols, axis=1) if cols else np.zeros((W,0), float)
    # Orthonormalize by QR
    if X.size:
        Q, _ = np.linalg.qr(X)
        return Q
    return X


def build_complex_basis(t: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Complex exponential basis expanded to real-imag blocks (cos/sin) same as Fourier."""
    # For now identical numerically to fourier; left as hook for future complex phasing
    return build_fourier_basis(t, periods)


def _safe_norm(v: np.ndarray, eps=1e-12) -> float:
    n = float(np.sqrt((v**2).sum()))
    return max(n, eps)


def build_biquat_basis(t: np.ndarray, periods: np.ndarray, sigma: float=0.8) -> np.ndarray:
    """
    Biquaternion-phase basis: e^{(a i + b j + c k) * theta} expanded to 4 real channels:
      cos(|v| theta) + u_i sin(|v| theta), u_j sin(|v| theta), u_k sin(|v| theta),
    where v = (a,b,c), u = v / |v|. We generate multiple harmonics over theta_k(t)=2π t / P_k,
    with a,b,c derived from smooth functions of theta to inject slight anisotropy.

    Implementation detail:
      a = 1
      b = ema(cos(theta), alpha=sigma)
      c = ema(sin(theta), alpha=sigma)
    Then stack all channels and QR-orthonormalize.
    """
    W = len(t)
    cols = []
    for P in periods:
        theta = 2*np.pi * t / float(P)
        # smooth carriers (inject weak modulation)
        b_raw = np.cos(theta)
        c_raw = np.sin(theta)
        b = ema1(b_raw, alpha=sigma if sigma>0 else 1.0)
        c = ema1(c_raw, alpha=sigma if sigma>0 else 1.0)
        a = np.ones_like(theta)
        v = np.stack([a,b,c], axis=1)  # (W,3)
        # use mean direction over window to keep basis stable
        vbar = v.mean(axis=0)
        nrm = _safe_norm(vbar)
        u = vbar / nrm  # 3-dim unit
        w = nrm  # effective frequency scale multiplier

        wt = w * theta
        cwt = np.cos(wt)
        swt = np.sin(wt)

        # 4 real channels per period
        s0 = cwt
        s1 = u[0] * swt
        s2 = u[1] * swt
        s3 = u[2] * swt

        cols += [s0, s1, s2, s3]

    X = np.stack(cols, axis=1) if cols else np.zeros((W,0), float)
    if X.size:
        Q, _ = np.linalg.qr(X)
        return Q
    return X


def ridge_solve(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression solution (X^T X + lam I)^{-1} X^T y."""
    XT = X.T
    G = XT @ X
    lamI = lam * np.eye(G.shape[0], dtype=float)
    beta = np.linalg.solve(G + lamI, XT @ y)
    return beta


def fetch_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance if ccxt is installed. Otherwise raise.
    Returns DataFrame with columns: time (pd.Timestamp), close (float).
    """
    if symbol.lower().endswith(".csv"):
        df = pd.read_csv(symbol)
        # try to normalize columns
        if "time" not in df.columns and "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        if "close" not in df.columns:
            # try typical column name
            for c in ["Close", "close_price", "price"]:
                if c in df.columns:
                    df = df.rename(columns={c: "close"})
                    break
        # parse time
        if np.issubdtype(df["time"].dtype, np.number):
            df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df[["time","close"]].dropna().reset_index(drop=True)
        return df

    if _CCXT is None:
        raise RuntimeError("ccxt is not installed; install `pip install ccxt` or pass a CSV path as symbol")

    ex = _CCXT.binance({"enableRateLimit": True})
    tf = interval
    data = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    # ccxt columns: [ts, open, high, low, close, volume]
    arr = np.array(data, dtype=float)
    ts = pd.to_datetime(arr[:,0].astype(np.int64), unit="ms", utc=True)
    close = arr[:,4]
    return pd.DataFrame({"time": ts, "close": close})


@dataclass
class EvalRow:
    time: str
    entry_idx: int
    compare_idx: int
    last_price: float
    pred_price: float
    future_price: float
    pred_delta: float
    true_delta: float
    pred_dir: int
    true_dir: int
    hold_ret: float
    correct_pred: int
    hold_up: int


def evaluate_symbol(symbol: str, interval: str, window: int, horizon: int,
                    minP: int, maxP: int, nP: int, sigma: float,
                    lam: float, limit: int, phase: str,
                    out_prefix: str = "") -> Tuple[pd.DataFrame, dict]:

    df = fetch_ohlcv(symbol, interval, limit)
    prices = df["close"].to_numpy(dtype=float)
    times = df["time"].astype(str).to_list()

    W = int(window)
    H = int(horizon)

    rows: List[EvalRow] = []

    periods = make_period_grid(minP, maxP, nP)

    for entry in range(W, len(prices)-H):
        y_hist = prices[entry-W:entry]           # window history
        y_future = prices[entry+H]               # compare to H ahead
        last_price = prices[entry-1]

        t = np.arange(W, dtype=float)

        # choose basis per phase
        if phase == "biquat":
            X = build_biquat_basis(t, periods, sigma=sigma)
        elif phase == "complex":
            X = build_complex_basis(t, periods)
        else:
            X = build_fourier_basis(t, periods)

        # target is history relative to last price or absolute?
        # We fit absolute level to keep consistent with user's logs.
        y = y_hist

        if X.size == 0:
            pred = last_price
        else:
            beta = ridge_solve(X, y, lam=lam)
            # forecast H steps ahead by extending theta with t_H = W-1 + H
            tH = (W-1) + H
            t_future = np.arange(W, dtype=float)  # reuse shape; synthesize via phase at index tH
            # Build a single-row basis vector at tH consistent with training design:
            def basis_row_at(t_scalar: float) -> np.ndarray:
                tt = np.arange(W, dtype=float)  # use same length to define a "phase context"
                if phase == "biquat":
                    # build basis for full window, then take last row's columns evaluated
                    Xf = build_biquat_basis(tt + (t_scalar - (W-1)), periods, sigma=sigma)
                elif phase == "complex":
                    Xf = build_complex_basis(tt + (t_scalar - (W-1)), periods)
                else:
                    Xf = build_fourier_basis(tt + (t_scalar - (W-1)), periods)
                # take the last row as the representative feature for t_scalar
                return Xf[-1,:] if Xf.size else np.zeros((0,), float)

            xH = basis_row_at(tH).reshape(1,-1)
            pred = float((xH @ beta).ravel()[0]) if xH.size else float(last_price)

        pred_price = pred
        future_price = y_future
        pred_delta = float(pred_price - last_price)
        true_delta = float(future_price - last_price)
        pred_dir = 1 if pred_delta > 0 else (-1 if pred_delta < 0 else 0)
        true_dir = 1 if true_delta > 0 else (-1 if true_delta < 0 else 0)
        hold_ret = (future_price - last_price) / last_price if last_price != 0 else 0.0

        rows.append(EvalRow(
            time=times[entry],
            entry_idx=entry-1,
            compare_idx=entry+H,
            last_price=float(last_price),
            pred_price=float(pred_price),
            future_price=float(future_price),
            pred_delta=float(pred_delta),
            true_delta=float(true_delta),
            pred_dir=int(pred_dir),
            true_dir=int(true_dir),
            hold_ret=float(hold_ret),
            correct_pred=int(1 if pred_dir == true_dir else 0),
            hold_up=int(1 if hold_ret > 0 else 0),
        ))

    out = pd.DataFrame([r.__dict__ for r in rows])

    # metrics
    hit_rate_pred = float((out["correct_pred"]==1).mean()) if len(out) else 0.0
    hit_rate_hold = float((out["hold_up"]==1).mean()) if len(out) else 0.0
    delta_hit = hit_rate_pred - hit_rate_hold
    corr_pred_true = float(np.corrcoef(out["pred_delta"], out["true_delta"])[0,1]) if len(out)>=2 else 0.0
    mae_price = float(np.abs(out["pred_price"] - out["future_price"]).mean()) if len(out) else 0.0
    # compute returns in bps-like (relative) terms
    with np.errstate(divide='ignore', invalid='ignore'):
        ret_pred = (out["pred_price"] - out["last_price"]) / out["last_price"]
        ret_true = (out["future_price"] - out["last_price"]) / out["last_price"]
    mae_return = float(np.abs(ret_pred - ret_true).mean()) if len(out) else 0.0

    summary = {
        "symbol": symbol,
        "phase": phase,
        "hit_rate_pred": hit_rate_pred,
        "hit_rate_hold": hit_rate_hold,
        "delta_hit": delta_hit,
        "corr_pred_true": corr_pred_true,
        "mae_price": mae_price,
        "mae_return": mae_return,
        "count": int(len(out)),
    }

    prefix = out_prefix if out_prefix else ""
    sym_sanit = symbol.replace("/","").replace(":","").replace("-","")
    out_csv = f"eval_h_{sym_sanit}.csv"
    out_json = f"sum_h_{sym_sanit}.json"
    out.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n--- HSTRATEGY vs HOLD ---")
    print(f"hit_rate_pred:  {hit_rate_pred:.6f}")
    print(f"hit_rate_hold:  {hit_rate_hold:.6f}")
    print(f"corr_pred_true: {corr_pred_true:.6f}")
    print(f"mae_price:      {mae_price:.6f}")
    print(f"mae_return:     {mae_return:.6f}")
    print(f"count:          {len(out)}\n")
    print(f"Uloženo CSV: {out_csv}")
    print(f"Uloženo summary: {out_json}\n")
    return out, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated list of symbols or CSV paths")
    ap.add_argument('--csv-time-col', default='time',
                    help='CSV column name for timestamps when --symbols is a CSV file')
    ap.add_argument('--csv-close-col', default='close',
                    help='CSV column name for close prices when --symbols is a CSV file')
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--minP", type=int, default=24)
    ap.add_argument("--maxP", type=int, default=480)
    ap.add_argument("--nP", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--phase", choices=["simple","complex","biquat"], default="biquat",
                    help="Basis to use: simple=Fourier cos/sin, complex=complex exp (real-imag), biquat=biquaternion phase (default).")
    ap.add_argument("--out", default="hbatch_summary.csv")
    args = ap.parse_args()

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    summaries = []
    for sym in syms:
        print(f"\n=== Running {sym} ===")
        df, summ = evaluate_symbol(
            symbol=sym, interval=args.interval, window=args.window, horizon=args.horizon,
            minP=args.minP, maxP=args.maxP, nP=args.nP, sigma=args.sigma, lam=args.lam,
            limit=args.limit, phase=args.phase, out_prefix=""
        )
        summaries.append(summ)

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(args.out, index=False)
    print(f"\nUloženo: {args.out}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
