#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Theta OrthoQR: weighted-QR orthogonal theta fit + baselines
#
# Tau(t) = t + i * psi(t), psi(t) either constant or EMA(|returns|) * scale
# Atoms: cos(omega * tau), sin(omega * tau)  (using complex tau inside cos/sin)
# Weighted LS via QR with weights w_t; global refit after each added mode (OMP+QR)
#
# NOTE: No external deps beyond numpy/pandas/requests.

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
try:
    import requests
except Exception:
    requests = None

# ---------- helpers ----------
def parse_interval_to_minutes(interval: str) -> int:
    interval = interval.strip().lower()
    num_str = "".join(ch for ch in interval if ch.isdigit())
    unit = "".join(ch for ch in interval if ch.isalpha())
    if not num_str or not unit:
        raise ValueError(f"Neplatný interval: {interval}")
    num = int(num_str)
    if unit == 'm': return num
    if unit == 'h': return num * 60
    if unit == 'd': return num * 60 * 24
    if unit == 'w': return num * 60 * 24 * 7
    raise ValueError(f"Neznámý interval: {interval}")

def parse_horizons(tokens: List[str], base_interval_minutes: int) -> List[int]:
    out: List[int] = []
    for t in tokens:
        t = t.strip().lower()
        if t.isdigit():
            out.append(int(t)); continue
        num_str = "".join(ch for ch in t if ch.isdigit())
        unit = "".join(ch for ch in t if ch.isalpha())
        if not num_str or not unit:
            raise ValueError(f"Neplatný horizon token: {t}")
        num = int(num_str)
        minutes = {'m': num, 'h': num*60, 'd': num*60*24, 'w': num*60*24*7}.get(unit)
        if minutes is None:
            raise ValueError(f"Neplatný horizon token: {t}")
        bars = max(1, round(minutes / base_interval_minutes))
        out.append(bars)
    out = sorted(set(out))
    return out

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    short2long = {"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}
    if any(c in df.columns for c in short2long):
        df.rename(columns={k:v for k,v in short2long.items() if k in df.columns}, inplace=True)
    alt = {"Timestamp":"timestamp","Time":"timestamp","open_time":"timestamp",
           "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
    for k,v in alt.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)
    req = ["timestamp","open","high","low","close","volume"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f"Chybí sloupce {missing}; mám: {list(df.columns)}")
    ts = pd.to_datetime(df["timestamp"], errors="coerce", unit="ms")
    if ts.isna().all():
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = ts
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp","close"]).sort_values("timestamp").reset_index(drop=True)

def fetch_klines_binance(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    if requests is None:
        raise RuntimeError("requests není k dispozici. Použij --csv.")
    url = "https://api.binance.com/api/v3/klines"
    max_batch = 1000
    remaining = int(limit)
    rows = []
    params = {"symbol": symbol.upper(), "interval": interval, "limit": min(max_batch, remaining)}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    batch = r.json()
    if not batch: raise RuntimeError("Binance vrátil prázdnou odpověď pro klines.")
    rows.extend(batch); remaining -= len(batch)
    end_time = int(batch[0][0]) - 1
    while remaining > 0:
        params = {"symbol": symbol.upper(), "interval": interval, "limit": min(max_batch, remaining), "endTime": end_time}
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()
        batch = r.json()
        if not batch: break
        rows.extend(batch); remaining -= len(batch)
        end_time = int(batch[0][0]) - 1
    rows.reverse()
    out = [{
        "timestamp": int(k[0]),
        "open": float(k[1]),
        "high": float(k[2]),
        "low":  float(k[3]),
        "close": float(k[4]),
        "volume": float(k[5]),
    } for k in rows]
    return normalize_ohlcv(pd.DataFrame(out))

def load_or_download(symbol: str, interval: str, limit: int, csv_path: str|None, outdir: Path) -> pd.DataFrame:
    if csv_path:
        return normalize_ohlcv(pd.read_csv(csv_path))
    cache = outdir / f"ohlcv_{symbol}_{interval}_{limit}.csv"
    if cache.exists():
        return normalize_ohlcv(pd.read_csv(cache))
    df = fetch_klines_binance(symbol, interval, limit)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df

# ---------- baselines ----------
def _lin_trend(y: np.ndarray):
    n = len(y)
    if n < 2: return 0.0, float(y[-1]) if n else 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x-xm)**2)
    if denom == 0: return 0.0, float(ym)
    a = np.sum((x-xm)*(y-ym))/denom
    b = ym - a*xm
    return float(a), float(b)

def forecast_raw(window_vals: np.ndarray, horizon: int) -> float:
    return float(window_vals[-1])

# ---------- theta atoms & weights ----------
def ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    if len(arr) == 0: return arr
    out = np.empty_like(arr, dtype=float)
    s = 0.0
    a = float(alpha)
    one_ma = float(1.0 - a)
    for i, v in enumerate(arr):
        if i == 0: s = float(v)
        else: s = a*float(v) + one_ma*s
        out[i] = s
    return out

def make_tau_and_weights(n: int, y: np.ndarray, psi_mode: str, psi_const: float, psi_scale: float, psi_ema: int, w_alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=float)
    # returns for psi construction
    dy = np.diff(y, prepend=y[0])
    absret = np.abs(dy) / max(1e-12, np.nanmean(np.abs(y)))
    if psi_mode == "const":
        psi = np.full(n, float(psi_const), dtype=float)
    elif psi_mode == "ema_absret":
        a = 2.0 / max(1, psi_ema + 1)
        psi = psi_scale * ema(absret, a)
    else:
        psi = np.zeros(n, dtype=float)
    # tau
    tau = t + 1j*psi
    # weights ~ 1 + alpha * ema_absret (proxy for |d tau/dt|)
    if w_alpha > 0.0:
        w = 1.0 + w_alpha * ema(absret, 2.0/max(1, psi_ema + 1))
    else:
        w = np.ones(n, dtype=float)
    return tau, w

def theta_columns(tau: np.ndarray, omega: float) -> np.ndarray:
    # Build real columns from complex tau: cos(omega * tau), sin(omega * tau)
    z = omega * tau
    c = np.cos(z)
    s = np.sin(z)
    # use real part for both columns (cos, sin with complex argument yield complex -> take real component)
    # Alternatively, could use [Re(c), Im(c)] etc., but we keep consistency with real target y.
    C = np.column_stack([np.real(c), np.real(s)])
    return C

# ---------- weighted QR solve ----------
def weighted_qr_solve(C: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float = 0.0) -> Tuple[np.ndarray, float]:
    # Normalize columns (in W-inner product) for stability
    Wsqrt = np.sqrt(w)
    Cw = C * Wsqrt[:,None]
    yw = y * Wsqrt
    # Ridge via augmented system
    if lam > 0.0:
        # Augment with sqrt(lam)*I
        d = C.shape[1]
        C_aug = np.vstack([Cw, np.sqrt(lam)*np.eye(d)])
        y_aug = np.concatenate([yw, np.zeros(d)])
        Q, R = np.linalg.qr(C_aug, mode='reduced')
        beta = np.linalg.solve(R, Q.T @ y_aug)
    else:
        Q, R = np.linalg.qr(Cw, mode='reduced')
        beta = np.linalg.solve(R, Q.T @ yw)
    resid = yw - Cw @ beta
    rss = float(resid @ resid)
    return beta, rss

# ---------- OMP+global refit ----------
def theta_ortho_qr_fit(y: np.ndarray, topn: int, omega_grid: np.ndarray, tau: np.ndarray, w: np.ndarray, lam: float) -> Tuple[List[Tuple[float,float,float]], float]:
    n = len(y)
    modes: List[Tuple[float,float,float]] = []
    # start with empty design
    C_sel = np.zeros((n,0))
    rss_best = float(np.dot((y*np.sqrt(w)), (y*np.sqrt(w))))  # initial weighted RSS
    for _ in range(max(1, topn)):
        best = None
        best_gain = 0.0
        for om in omega_grid:
            # candidate columns
            Cc = theta_columns(tau, om)  # n x 2
            # try augment and solve globally
            C_try = np.hstack([C_sel, Cc])
            beta_try, rss_try = weighted_qr_solve(C_try, y, w, lam=lam)
            gain = rss_best - rss_try
            if gain > best_gain:
                best_gain = gain
                best = (om, Cc, beta_try, rss_try, C_try)
        if best is None or best_gain <= 0.0:
            break
        om, Cc, beta_try, rss_try, C_try = best
        # accept
        C_sel = C_try
        rss_best = rss_try
        # last two betas correspond to this mode
        Acoef, Bcoef = float(beta_try[-2]), float(beta_try[-1])
        modes.append((om, Acoef, Bcoef))
    return modes, rss_best

# ---------- Forecast ----------
def forecast_theta_orthoqr(window_vals: np.ndarray, horizon: int,
                           topn: int, zero_pad: int, min_period_bars: int, lam: float,
                           psi_mode: str, psi_const: float, psi_scale: float, psi_ema: int, w_alpha: float,
                           damping: float, shrink: float) -> float:
    y = window_vals.astype(float)
    n = len(y)
    # detrend
    a,b = _lin_trend(y)
    trend = a*np.arange(n)+b
    r = y - trend

    # tau & weights
    tau, w = make_tau_and_weights(n, y, psi_mode, psi_const, psi_scale, psi_ema, w_alpha)

    # frequency grid (coarse, real-time friendly)
    N = int(n * max(1, zero_pad))
    freqs = np.fft.rfftfreq(N, d=1.0)
    if min_period_bars > 0:
        freqs = freqs[(freqs > 0) & ((1.0/np.maximum(freqs, 1e-12)) >= min_period_bars)]
    omega_grid = 2.0*np.pi*freqs
    if omega_grid.size == 0:
        return float(y[-1])

    # OMP + global weighted QR
    modes, _ = theta_ortho_qr_fit(r, topn=topn, omega_grid=omega_grid, tau=tau, w=w, lam=lam)

    # forecast at t_f = (n-1)+h using same tau model:
    t_f = (n - 1) + horizon
    # extrapolate psi deterministically: use last psi value (hold) for simplicity
    tau_f = t_f + 1j * np.real(tau[-1].imag)  # hold-last psi
    synth = 0.0
    for (om, Acoef, Bcoef) in modes:
        synth += Acoef*np.cos(om * tau_f) + Bcoef*np.sin(om * tau_f)
    if shrink > 0.0:
        synth *= (1.0 - shrink)
    if damping > 0.0:
        synth *= float(np.exp(-damping * horizon))
    y_hat = synth + (a*(n-1+horizon)+b)
    return float(np.real(y_hat))

# ---------- rolling ----------
def build_roll_forecast(df: pd.DataFrame, variants: List[str], window: int, horizons: List[int],
                        horizon_alpha: float,
                        ortho_topn: int, ortho_zero_pad: int, ortho_min_period_bars: int, ortho_lam: float,
                        theta_psi_mode: str, theta_psi_const: float, theta_psi_scale: float, theta_psi_ema: int, theta_w_alpha: float,
                        ortho_damping: float, ortho_shrink: float
                        ) -> Tuple[List[Dict], Dict]:
    ts = pd.to_datetime(df["timestamp"])
    closes = df["close"].astype(float).to_numpy()
    n = len(closes)
    start = window
    end = n - max(horizons)
    if end <= start:
        raise RuntimeError(f"Nedostatek dat: n={n}, window={window}, horizons={horizons}")
    rows: List[Dict] = []
    for t in range(start, end):
        w = closes[t-window:t]
        row = {"timestamp": ts.iloc[t].isoformat(), "close": float(closes[t])}
        for H in horizons:
            if "raw" in variants:
                v = forecast_raw(w, H)
                row[f"pred_raw_h{H}"] = float(horizon_alpha*v + (1.0-horizon_alpha)*row["close"])
            if "thetaOrthoQR" in variants:
                v = forecast_theta_orthoqr(
                    w, H,
                    topn=ortho_topn, zero_pad=ortho_zero_pad, min_period_bars=ortho_min_period_bars, lam=ortho_lam,
                    psi_mode=theta_psi_mode, psi_const=theta_psi_const, psi_scale=theta_psi_scale, psi_ema=theta_psi_ema, w_alpha=theta_w_alpha,
                    damping=ortho_damping, shrink=ortho_shrink
                )
                row[f"pred_thetaOrthoQR_h{H}"] = float(horizon_alpha*v + (1.0-horizon_alpha)*row["close"])
        rows.append(row)
    debug = {"params":{
        "window":window,"horizons":horizons,"horizon_alpha":horizon_alpha,
        "thetaOrthoQR":{
            "topn":ortho_topn,"zero_pad":ortho_zero_pad,"min_period_bars":ortho_min_period_bars,
            "lam":ortho_lam,
            "psi_mode":theta_psi_mode,"psi_const":theta_psi_const,"psi_scale":theta_psi_scale,"psi_ema":theta_psi_ema,"w_alpha":theta_w_alpha,
            "damping":ortho_damping,"shrink":ortho_shrink
        }},
        "n_rows": len(rows)}
    return rows, debug

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Theta OrthoQR (weighted-QR orthogonal theta fit)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--variants", nargs="+", default=["raw","thetaOrthoQR"])
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--horizons", nargs="+", default=["1h"])
    p.add_argument("--horizon-alpha", type=float, default=1.0, dest="horizon_alpha")
    # thetaOrthoQR
    p.add_argument("--ortho-topn", type=int, default=1, dest="ortho_topn")
    p.add_argument("--ortho-zero-pad", type=int, default=4, dest="ortho_zero_pad")
    p.add_argument("--ortho-min-period-bars", type=int, default=24, dest="ortho_min_period_bars")
    p.add_argument("--ortho-lam", type=float, default=1e-5, dest="ortho_lam")
    p.add_argument("--theta-psi-mode", type=str, default="const", dest="theta_psi_mode", choices=["const","ema_absret"])
    p.add_argument("--theta-psi-const", type=float, default=0.0, dest="theta_psi_const")
    p.add_argument("--theta-psi-scale", type=float, default=1.0, dest="theta_psi_scale")
    p.add_argument("--theta-psi-ema", type=int, default=32, dest="theta_psi_ema")
    p.add_argument("--theta-w-alpha", type=float, default=0.0, dest="theta_w_alpha", help="váhová citlivost na EMA(|ret|); 0=jednotkové váhy")
    p.add_argument("--ortho-damping", type=float, default=0.0, dest="ortho_damping")
    p.add_argument("--ortho-shrink", type=float, default=0.0, dest="ortho_shrink")
    p.add_argument("--outdir", type=str, default="reports_forecast")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base_minutes = parse_interval_to_minutes(args.interval)
    horizons = parse_horizons(args.horizons, base_minutes)

    if args.csv:
        df = normalize_ohlcv(pd.read_csv(args.csv))
    else:
        df = load_or_download(args.symbol, args.interval, args.limit, None, outdir)

    rows, debug = build_roll_forecast(
        df, args.variants, args.window, horizons, args.horizon_alpha,
        args.ortho_topn, args.ortho_zero_pad, args.ortho_min_period_bars, args.ortho_lam,
        args.theta_psi_mode, args.theta_psi_const, args.theta_psi_scale, args.theta_psi_ema, args.theta_w_alpha,
        args.ortho_damping, args.ortho_shrink
    )

    base = f"{args.symbol}_{args.interval}_win{args.window}_h{'-'.join(map(str,horizons))}_pure{int(args.horizon_alpha==1.0)}"
    csv_path = outdir / f"forecast_{base}.csv"
    log_path = outdir / f"forecast_{base}.json"

    if rows:
        cols = ["timestamp","close"] + sorted([k for k in rows[0].keys() if k.startswith('pred_')])
        pd.DataFrame(rows)[cols].to_csv(csv_path, index=False)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "debug": debug}, f, ensure_ascii=False, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {log_path}")

if __name__ == "__main__":
    main()
