#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Theta Forecast Runner with EKF/PLL:
#   - thetaPLL: EKF per frequency mode [A, B, phi, omega, domega] + Kalman trend (level+slope)
#   - Also keeps: raw, fft, thetaDyn2, thetaHybrid (optional to compare if present earlier)
#
# NOTE: This file is standalone; only uses numpy/pandas (no extra deps).

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

# ---------- helpers: intervals & horizons ----------
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

# ---------- data loading ----------
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

# ---------- helpers ----------
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

def forecast_fft(window_vals: np.ndarray, horizon: int, topn: int) -> float:
    y = window_vals.astype(float)
    n = len(y)
    a,b = _lin_trend(y)
    trend = a*np.arange(n)+b
    y_d = y - trend
    Y = np.fft.rfft(y_d)
    freqs = np.fft.rfftfreq(n, d=1.0)
    amps = np.abs(Y)
    idxs = np.argsort(amps)[::-1][:topn]
    t = n + horizon
    synth = 0.0+0.0j
    for k in idxs:
        synth += Y[k] * np.exp(2j*np.pi*freqs[k]*t)
    y_hat = (synth.real)/n
    return float(y_hat + (a*(n+horizon)+b))

# ---------- selection & initialisation ----------
def _select_top_freqs(y: np.ndarray, m_top: int, min_period_bars: int = 0):
    n = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0)
    amps = np.abs(Y)
    if len(amps) > 0:
        amps[0] = 0.0
    idxs = np.argsort(amps)[::-1]
    out = []
    for k in idxs:
        if len(out) >= max(1, m_top): break
        f = freqs[k]
        if min_period_bars > 0 and f > 0:
            period = 1.0/f
            if period < min_period_bars:
                continue
        out.append((k, 2.0*np.pi*f, Y[k]))
    return out  # list of tuples: (bin_index, omega, complex_amp)

def _init_mode_from_fft(k: int, Yk: complex, n: int):
    # Initial phase from FFT bin argument. Scale A,B by 2/n (cosine series) for rough start.
    phi0 = float(np.angle(Yk))
    # Initial amplitudes proportional to magnitude; distribute between cos/sin via phase
    R = (2.0 * np.abs(Yk)) / max(1.0, n)
    A0 = R * np.cos(phi0)
    B0 = R * np.sin(phi0)
    return A0, B0, phi0

# ---------- Kalman trend (level + slope) ----------
def _kalman_trend(y: np.ndarray, qL: float, qS: float, r: float):
    n = len(y)
    x = np.zeros((2,1)); P = np.eye(2)*1.0
    A = np.array([[1.0,1.0],[0.0,1.0]])
    Q = np.array([[qL,0.0],[0.0,qS]])
    H = np.array([[1.0,0.0]])
    R = float(r)
    levels = np.zeros(n)
    for k in range(n):
        x = A @ x
        P = A @ P @ A.T + Q
        yk = np.array([[y[k]]])
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ (yk - H @ x)
        P = (np.eye(2) - K @ H) @ P
        levels[k] = float(x[0,0])
    return x.reshape(-1), levels  # last [level, slope]

# ---------- EKF per-mode (A,B,phi,omega,domega) ----------
def _ekf_modes(residual: np.ndarray, modes: List[Tuple[int, float, complex]],
               qA: float, qB: float, qphi: float, qomega: float, qdomega: float, R_meas: float):
    """
    EKF over stacked state [A1,B1,phi1,omega1,domega1, A2,B2,phi2,omega2,domega2, ...]
    Nonlinear evolution:
        A' = A + wA
        B' = B + wB
        phi' = phi + omega + 0.5*domega + wphi
        omega' = omega + domega + womega
        domega' = domega + w_domega
    Measurement:
        y = sum_i (A_i cos(phi_i) + B_i sin(phi_i)) + v
    """
    y = residual.astype(float)
    n = len(y)
    m = len(modes)
    if m == 0 or n == 0:
        return np.zeros(0)
    dim = 5*m
    # init state
    x = np.zeros((dim,1))
    P = np.eye(dim) * 1.0
    # set initial guesses from FFT
    for i,(k, omega_i, Yk) in enumerate(modes):
        A0,B0,phi0 = _init_mode_from_fft(k, Yk, n)
        idx = 5*i
        x[idx+0,0] = A0
        x[idx+1,0] = B0
        x[idx+2,0] = phi0
        x[idx+3,0] = omega_i
        x[idx+4,0] = 0.0  # initial frequency acceleration
    # process noise
    Qi = np.diag([qA, qB, qphi, qomega, qdomega])
    Q = np.kron(np.eye(m), Qi)
    R = float(R_meas)

    for t in range(n):
        # --- predict ---
        # x' = f(x)
        x_pred = x.copy()
        for i in range(m):
            base = 5*i
            A = x[base+0,0]; B = x[base+1,0]; phi = x[base+2,0]; om = x[base+3,0]; dom = x[base+4,0]
            A_p = A
            B_p = B
            phi_p = phi + om + 0.5*dom
            om_p = om + dom
            dom_p = dom
            x_pred[base+0,0] = A_p
            x_pred[base+1,0] = B_p
            x_pred[base+2,0] = phi_p
            x_pred[base+3,0] = om_p
            x_pred[base+4,0] = dom_p
        # F = Jacobian of f
        F = np.eye(dim)
        for i in range(m):
            base = 5*i
            F[base+2, base+2] = 1.0  # dphi'/dphi
            F[base+2, base+3] = 1.0  # dphi'/dom
            F[base+2, base+4] = 0.5  # dphi'/ddom
            F[base+3, base+3] = 1.0  # dom'/dom
            F[base+3, base+4] = 1.0  # dom'/ddom
            # A',B',ddom' jacobians are identity already
        P_pred = F @ P @ F.T + Q

        # --- update ---
        # measurement: y_t = sum_i (A_i cos(phi_i) + B_i sin(phi_i)) + v
        H = np.zeros((1, dim))
        yhat = 0.0
        for i in range(m):
            base = 5*i
            A = x_pred[base+0,0]; B = x_pred[base+1,0]; phi = x_pred[base+2,0]
            c = np.cos(phi); s = np.sin(phi)
            yhat += A * c + B * s
            H[0, base+0] = c           # d y / d A
            H[0, base+1] = s           # d y / d B
            H[0, base+2] = -A*s + B*c  # d y / d phi
            # d y / d omega, d y / d domega are zero in measurement model
        yk = np.array([[y[t]]])
        innov = yk - np.array([[yhat]])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ innov
        P = (np.eye(dim) - K @ H) @ P_pred

        # keep phi in (-pi, pi] to avoid numeric blowup
        for i in range(m):
            base = 5*i
            phi = x[base+2,0]
            # wrap
            x[base+2,0] = (phi + np.pi) % (2*np.pi) - np.pi

    return x.reshape(-1)

def _propagate_modes(x_modes: np.ndarray, horizon: int) -> np.ndarray:
    """Propagate EKF mode states forward h steps deterministically (no noise)."""
    x = x_modes.copy()
    m = len(x)//5
    for _ in range(horizon):
        for i in range(m):
            base = 5*i
            A = x[base+0]; B = x[base+1]; phi = x[base+2]; om = x[base+3]; dom = x[base+4]
            # evolution
            phi = phi + om + 0.5*dom
            om = om + dom
            # keep phi wrapped
            phi = (phi + np.pi) % (2*np.pi) - np.pi
            # write back
            x[base+2] = phi
            x[base+3] = om
            # A,B,dom stay same
    return x

def _modes_to_signal(x_modes: np.ndarray) -> float:
    m = len(x_modes)//5
    y = 0.0
    for i in range(m):
        base = 5*i
        A = x_modes[base+0]; B = x_modes[base+1]; phi = x_modes[base+2]
        y += A*np.cos(phi) + B*np.sin(phi)
    return float(y)

# ---------- Forecasts ----------
def forecast_theta_pll(window_vals: np.ndarray, horizon: int,
                       # trend
                       qL: float, qS: float, r_trend: float,
                       # mode selection
                       m_top: int, min_period_bars: int,
                       # EKF noises
                       qA: float, qB: float, qphi: float, qomega: float, qdomega: float,
                       R_meas: float,
                       # regularisation
                       gamma: float, shrink: float) -> float:
    y = window_vals.astype(float)
    n = len(y)
    # 1) Trend Kalman
    xT, levels = _kalman_trend(y, qL=qL, qS=qS, r=r_trend)
    level, slope = float(xT[0]), float(xT[1])
    level_future = level + slope*float(horizon)
    residual = y - levels
    if n < 32:
        return float(level_future)
    # 2) Init modes from FFT (on residuals)
    modes = _select_top_freqs(residual, m_top=m_top, min_period_bars=min_period_bars)
    if len(modes) == 0:
        return float(level_future)
    # 3) EKF estimation on residual
    x_modes = _ekf_modes(residual, modes, qA, qB, qphi, qomega, qdomega, R_meas)
    # optional shrink
    if shrink > 0.0 and x_modes.size > 0:
        for i in range(len(x_modes)//5):
            base = 5*i
            x_modes[base+0] *= (1.0 - shrink)
            x_modes[base+1] *= (1.0 - shrink)
    # 4) Propagate to t+h and evaluate
    x_future = _propagate_modes(x_modes, horizon)
    res_future = _modes_to_signal(x_future)
    # damping with horizon
    if gamma > 0.0:
        res_future *= float(np.exp(-gamma * horizon))
    return float(level_future + res_future)

# ---------- rolling engine ----------
def build_roll_forecast(df: pd.DataFrame, variants: List[str], window: int, horizons: List[int],
                        horizon_alpha: float, fft_topn: int,
                        # EKF/PLL hyperparams
                        pll_qL: float, pll_qS: float, pll_r: float,
                        pll_topn: int, pll_min_period_bars: int,
                        pll_qA: float, pll_qB: float, pll_qphi: float, pll_qomega: float, pll_qdomega: float,
                        pll_R: float, pll_gamma: float, pll_shrink: float
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
            if "fft" in variants:
                v = forecast_fft(w, H, topn=fft_topn)
                row[f"pred_fft_h{H}"] = float(horizon_alpha*v + (1.0-horizon_alpha)*row["close"])
            if "thetaPLL" in variants:
                v = forecast_theta_pll(
                    w, H,
                    pll_qL, pll_qS, pll_r,
                    pll_topn, pll_min_period_bars,
                    pll_qA, pll_qB, pll_qphi, pll_qomega, pll_qdomega,
                    pll_R, pll_gamma, pll_shrink
                )
                row[f"pred_thetaPLL_h{H}"] = float(horizon_alpha*v + (1.0-horizon_alpha)*row["close"])
        rows.append(row)
    debug = {"params":{
        "window":window,"horizons":horizons,"horizon_alpha":horizon_alpha,
        "fft_topn":fft_topn,
        "thetaPLL":{
            "qL":pll_qL,"qS":pll_qS,"r":pll_r,
            "topn":pll_topn,"min_period_bars":pll_min_period_bars,
            "qA":pll_qA,"qB":pll_qB,"qphi":pll_qphi,"qomega":pll_qomega,"qdomega":pll_qdomega,
            "R":pll_R,"gamma":pll_gamma,"shrink":pll_shrink
        }},
        "n_rows": len(rows)}
    return rows, debug

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Theta rolling forecast with EKF/PLL (thetaPLL)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--variants", nargs="+", default=["raw","fft","thetaPLL"])
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--horizons", nargs="+", default=["1h"])
    p.add_argument("--horizon-alpha", type=float, default=1.0, dest="horizon_alpha")
    p.add_argument("--fft-topn", type=int, default=8, dest="fft_topn")
    # PLL (trend)
    p.add_argument("--pll-qL", type=float, default=1e-4, dest="pll_qL")
    p.add_argument("--pll-qS", type=float, default=1e-5, dest="pll_qS")
    p.add_argument("--pll-r", type=float, default=2e-2, dest="pll_r")
    # PLL (modes)
    p.add_argument("--pll-topn", type=int, default=1, dest="pll_topn")
    p.add_argument("--pll-min-period-bars", type=int, default=24, dest="pll_min_period_bars")
    # EKF noises
    p.add_argument("--pll-qA", type=float, default=1e-5, dest="pll_qA")
    p.add_argument("--pll-qB", type=float, default=1e-5, dest="pll_qB")
    p.add_argument("--pll-qphi", type=float, default=1e-6, dest="pll_qphi")
    p.add_argument("--pll-qomega", type=float, default=1e-6, dest="pll_qomega")
    p.add_argument("--pll-qdomega", type=float, default=1e-8, dest="pll_qdomega")
    p.add_argument("--pll-R", type=float, default=1e-2, dest="pll_R")
    # regularisation
    p.add_argument("--pll-gamma", type=float, default=0.0, dest="pll_gamma", help="Horizon damping of residual forecast")
    p.add_argument("--pll-shrink", type=float, default=0.0, dest="pll_shrink", help="Amplitude shrinkage 0..1")
    p.add_argument("--outdir", type=str, default="reports_forecast")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base_minutes = parse_interval_to_minutes(args.interval)
    horizons = parse_horizons(args.horizons, base_minutes)

    # load data
    if args.csv:
        df = normalize_ohlcv(pd.read_csv(args.csv))
    else:
        df = load_or_download(args.symbol, args.interval, args.limit, None, outdir)

    # roll
    rows, debug = build_roll_forecast(
        df, args.variants, args.window, horizons, args.horizon_alpha, args.fft_topn,
        args.pll_qL, args.pll_qS, args.pll_r,
        args.pll_topn, args.pll_min_period_bars,
        args.pll_qA, args.pll_qB, args.pll_qphi, args.pll_qomega, args.pll_qdomega,
        args.pll_R, args.pll_gamma, args.pll_shrink
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
