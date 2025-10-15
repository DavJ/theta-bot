#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Theta Forecast Runner with Guarded PLL (thetaPLLGuard):
#   - Robust PLL: EKF per frequency mode [A, B, phi, omega, domega] + Kalman trend (level+slope)
#   - Guards: innovation gating, R-boost, omega/domega clamps, auto-FFT reseed, post-reset damping

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

# ---------- basic helpers ----------
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

def _hann(n: int) -> np.ndarray:
    if n < 1: return np.ones(0)
    i = np.arange(n, dtype=float)
    return 0.5 - 0.5*np.cos(2.0*np.pi*i/(n-1))

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
    return out

def _init_mode_from_fft(k: int, Yk: complex, n: int):
    phi0 = float(np.angle(Yk))
    R = (2.0 * np.abs(Yk)) / max(1.0, n)
    A0 = R * np.cos(phi0)
    B0 = R * np.sin(phi0)
    return A0, B0, phi0

# ---------- Kalman trend ----------
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
    return x.reshape(-1), levels

# ---------- Guarded EKF per-mode ----------
def _ekf_modes_guard(residual: np.ndarray, modes: List[Tuple[int, float, complex]],
                     qA: float, qB: float, qphi: float, qomega: float, qdomega: float, R_meas: float,
                     guard_k: float, guard_boost: float, guard_max_misses: int,
                     reseed_enabled: bool, reseed_window_vals: np.ndarray|None, reseed_topn: int,
                     min_period_bars: int,
                     dw_max: float, d2w_max: float):
    y = residual.astype(float)
    n = len(y)
    m = len(modes)
    if m == 0 or n == 0:
        return np.zeros(0), 0

    dim = 5*m
    x = np.zeros((dim,1))
    P = np.eye(dim) * 1.0

    for i,(k, omega_i, Yk) in enumerate(modes):
        A0,B0,phi0 = _init_mode_from_fft(k, Yk, n)
        idx = 5*i
        x[idx+0,0] = A0
        x[idx+1,0] = B0
        x[idx+2,0] = phi0
        x[idx+3,0] = omega_i
        x[idx+4,0] = 0.0

    Qi = np.diag([qA, qB, qphi, qomega, qdomega])
    Q = np.kron(np.eye(m), Qi)
    R0 = float(R_meas)

    miss_count = 0
    post_reset = 0

    for t in range(n):
        # predict
        x_pred = x.copy()
        for i in range(m):
            base = 5*i
            A = x[base+0,0]; B = x[base+1,0]; phi = x[base+2,0]; om = x[base+3,0]; dom = x[base+4,0]
            A_p = A
            B_p = B
            phi_p = phi + om + 0.5*dom
            om_p = om + dom
            dom_p = dom
            # wrap phi
            phi_p = (phi_p + np.pi) % (2*np.pi) - np.pi
            # write
            x_pred[base+0,0] = A_p
            x_pred[base+1,0] = B_p
            x_pred[base+2,0] = phi_p
            x_pred[base+3,0] = om_p
            x_pred[base+4,0] = dom_p

        F = np.eye(dim)
        for i in range(m):
            base = 5*i
            F[base+2, base+2] = 1.0
            F[base+2, base+3] = 1.0
            F[base+2, base+4] = 0.5
            F[base+3, base+3] = 1.0
            F[base+3, base+4] = 1.0
        P_pred = F @ P @ F.T + Q

        # measurement
        H = np.zeros((1, dim))
        yhat = 0.0
        for i in range(m):
            base = 5*i
            A = x_pred[base+0,0]; B = x_pred[base+1,0]; phi = x_pred[base+2,0]
            c = np.cos(phi); s = np.sin(phi)
            yhat += A*c + B*s
            H[0, base+0] = c
            H[0, base+1] = s
            H[0, base+2] = -A*s + B*c
        yk = np.array([[y[t]]])
        innov = yk - np.array([[yhat]])
        S = H @ P_pred @ H.T + R0
        z2 = float(innov.squeeze()**2 / S.squeeze())

        # guard: large innovation
        if z2 > guard_k*guard_k:
            miss_count += 1
            R_eff = R0 * guard_boost
            # Update with boosted R to avoid destabilization
            Sg = H @ P_pred @ H.T + R_eff
            K = P_pred @ H.T @ np.linalg.inv(Sg)
            x = x_pred + K @ innov
            P = (np.eye(dim) - K @ H) @ P_pred
        else:
            miss_count = 0
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ innov
            P = (np.eye(dim) - K @ H) @ P_pred

        # clamp frequency dynamics
        for i in range(m):
            base = 5*i
            # retrieve previous omega for delta clamp
            # Using x_pred (pre-update) values for reference
            om_prev = x_pred[base+3,0]
            dom = x[base+4,0]
            om = x[base+3,0]
            # clamp domega
            dom = float(np.clip(dom, -d2w_max, d2w_max))
            x[base+4,0] = dom
            # clamp omega change
            om = float(np.clip(om, om_prev - dw_max, om_prev + dw_max))
            x[base+3,0] = om
            # wrap phi again (after update)
            phi = x[base+2,0]
            x[base+2,0] = (phi + np.pi) % (2*np.pi) - np.pi

        # auto-reseed if too many misses
        if reseed_enabled and miss_count >= guard_max_misses and reseed_window_vals is not None:
            # reseed using FFT on recent window
            rw = reseed_window_vals.astype(float)
            n_w = len(rw)
            a,b = _lin_trend(rw)
            trend = a*np.arange(n_w)+b
            r = rw - trend
            modes_new = _select_top_freqs(r, m_top=m, min_period_bars=min_period_bars)
            if len(modes_new) > 0:
                for i,(k2, omega2, Yk2) in enumerate(modes_new):
                    A0,B0,phi0 = _init_mode_from_fft(k2, Yk2, n_w)
                    base = 5*i
                    x[base+0,0] = A0*0.5  # slightly shrunk
                    x[base+1,0] = B0*0.5
                    x[base+2,0] = ((phi0 + np.pi) % (2*np.pi)) - np.pi
                    x[base+3,0] = omega2
                    x[base+4,0] = 0.0
                miss_count = 0
                post_reset += 1  # flag that a reset occurred

    return x.reshape(-1), post_reset

def _propagate_modes(x_modes: np.ndarray, horizon: int, dw_max: float, d2w_max: float) -> np.ndarray:
    x = x_modes.copy()
    m = len(x)//5
    for _ in range(horizon):
        for i in range(m):
            base = 5*i
            A = x[base+0]; B = x[base+1]; phi = x[base+2]; om = x[base+3]; dom = x[base+4]
            phi = phi + om + 0.5*dom
            om_next = om + dom
            # clamp domega
            dom = float(np.clip(dom, -d2w_max, d2w_max))
            # clamp omega change
            om_next = float(np.clip(om_next, om - dw_max, om + dw_max))
            om = om_next
            phi = (phi + np.pi) % (2*np.pi) - np.pi
            x[base+2] = phi
            x[base+3] = om
            x[base+4] = dom
    return x

def _modes_to_signal(x_modes: np.ndarray) -> float:
    m = len(x_modes)//5
    y = 0.0
    for i in range(m):
        base = 5*i
        A = x_modes[base+0]; B = x_modes[base+1]; phi = x_modes[base+2]
        y += A*np.cos(phi) + B*np.sin(phi)
    return float(y)

# ---------- Forecast ----------
def forecast_theta_pll_guard(window_vals: np.ndarray, horizon: int,
                             # trend
                             qL: float, qS: float, r_trend: float,
                             # modes selection
                             m_top: int, min_period_bars: int,
                             # EKF noises
                             qA: float, qB: float, qphi: float, qomega: float, qdomega: float,
                             R_meas: float,
                             # guards
                             guard_k: float, guard_boost: float, guard_max_misses: int,
                             reseed_window: int, reseed_topn: int,
                             dw_max: float, d2w_max: float,
                             # regularisation
                             gamma: float, shrink: float,
                             post_gamma: float, post_shrink: float, post_len: int) -> float:
    y = window_vals.astype(float)
    n = len(y)
    # trend
    xT, levels = _kalman_trend(y, qL=qL, qS=qS, r=r_trend)
    level, slope = float(xT[0]), float(xT[1])
    level_future = level + slope*float(horizon)
    residual = y - levels
    if n < 32:
        return float(level_future)
    # FFT init
    modes = _select_top_freqs(residual, m_top=m_top, min_period_bars=min_period_bars)
    if len(modes) == 0:
        return float(level_future)
    # reseed window
    reseed_vals = None
    if reseed_window > 0 and reseed_window < n:
        reseed_vals = y[-reseed_window:]
    # EKF with guards
    x_modes, post_reset = _ekf_modes_guard(
        residual, modes,
        qA, qB, qphi, qomega, qdomega, R_meas,
        guard_k, guard_boost, guard_max_misses,
        reseed_enabled=(reseed_window>0), reseed_window_vals=resid(residual, reseed_window) if reseed_window>0 else None, reseed_topn=reseed_topn,
        min_period_bars=min_period_bars,
        dw_max=dw_max, d2w_max=d2w_max
    )
    # propagate
    x_future = _propagate_modes(x_modes, horizon, dw_max=dw_max, d2w_max=d2w_max)
    res_future = _modes_to_signal(x_future)
    # shrink/damp (global)
    if shrink > 0.0:
        res_future *= (1.0 - shrink)
    if gamma > 0.0:
        res_future *= float(np.exp(-gamma * horizon))
    # post-reset damping
    if post_reset > 0 and post_len > 0:
        res_future *= float(np.exp(-post_gamma * horizon))
        res_future *= (1.0 - post_shrink)
    return float(level_future + res_future)

def resid(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 0 or w > len(arr):
        return arr.copy()
    return arr[-w:].copy()

# ---------- Rolling engine ----------
def build_roll_forecast(df: pd.DataFrame, variants: List[str], window: int, horizons: List[int],
                        horizon_alpha: float,
                        # thetaPLLGuard params
                        qL: float, qS: float, r_trend: float,
                        topn: int, min_period_bars: int,
                        qA: float, qB: float, qphi: float, qomega: float, qdomega: float, R_meas: float,
                        guard_k: float, guard_boost: float, guard_max_misses: int,
                        reseed_window: int, reseed_topn: int,
                        dw_max: float, d2w_max: float,
                        gamma: float, shrink: float, post_gamma: float, post_shrink: float, post_len: int
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
            if "thetaPLLGuard" in variants:
                v = forecast_theta_pll_guard(
                    w, H,
                    qL, qS, r_trend,
                    topn, min_period_bars,
                    qA, qB, qphi, qomega, qdomega, R_meas,
                    guard_k, guard_boost, guard_max_misses,
                    reseed_window, topn,
                    dw_max, d2w_max,
                    gamma, shrink, post_gamma, post_shrink, post_len
                )
                row[f"pred_thetaPLLGuard_h{H}"] = float(horizon_alpha*v + (1.0-horizon_alpha)*row["close"])
        rows.append(row)
    debug = {"params":{
        "window":window,"horizons":horizons,"horizon_alpha":horizon_alpha,
        "thetaPLLGuard":{
            "trend":{"qL":qL,"qS":qS,"r":r_trend},
            "modes":{"topn":topn,"min_period_bars":min_period_bars},
            "ekf":{"qA":qA,"qB":qB,"qphi":qphi,"qomega":qomega,"qdomega":qdomega,"R":R_meas},
            "guard":{"k":guard_k,"boost":guard_boost,"max_misses":guard_max_misses},
            "reseed":{"window":reseed_window,"topn":reseed_topn},
            "limits":{"dw_max":dw_max,"d2w_max":d2w_max},
            "reg":{"gamma":gamma,"shrink":shrink,"post_gamma":post_gamma,"post_shrink":post_shrink,"post_len":post_len}
        }},
        "n_rows": len(rows)}
    return rows, debug

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Theta rolling forecast with Guarded PLL (thetaPLLGuard)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--variants", nargs="+", default=["raw","thetaPLLGuard"])
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--horizons", nargs="+", default=["1h"])
    p.add_argument("--horizon-alpha", type=float, default=1.0, dest="horizon_alpha")
    # trend
    p.add_argument("--pll-qL", type=float, default=2e-5, dest="qL")
    p.add_argument("--pll-qS", type=float, default=5e-7, dest="qS")
    p.add_argument("--pll-r", type=float, default=2e-2, dest="r_trend")
    # modes
    p.add_argument("--pll-topn", type=int, default=1, dest="topn")
    p.add_argument("--pll-min-period-bars", type=int, default=24, dest="min_period_bars")
    # ekf noises
    p.add_argument("--pll-qA", type=float, default=2e-6, dest="qA")
    p.add_argument("--pll-qB", type=float, default=2e-6, dest="qB")
    p.add_argument("--pll-qphi", type=float, default=1e-7, dest="qphi")
    p.add_argument("--pll-qomega", type=float, default=5e-8, dest="qomega")
    p.add_argument("--pll-qdomega", type=float, default=1e-10, dest="qdomega")
    p.add_argument("--pll-R", type=float, default=1e-2, dest="R_meas")
    # guards
    p.add_argument("--pll-guard-k", type=float, default=3.0, dest="guard_k")
    p.add_argument("--pll-guard-boost", type=float, default=10.0, dest="guard_boost")
    p.add_argument("--pll-guard-max-misses", type=int, default=3, dest="guard_max_misses")
    # reseed
    p.add_argument("--pll-reseed-window", type=int, default=96, dest="reseed_window", help="Počet posledních barů pro FFT reseed (rezidua). 0=off")
    p.add_argument("--pll-reseed-topn", type=int, default=1, dest="reseed_topn")
    # limits
    p.add_argument("--pll-dw-max", type=float, default=0.02, dest="dw_max", help="Max |Δomega| per step")
    p.add_argument("--pll-d2w-max", type=float, default=0.001, dest="d2w_max", help="Max |domega| per step")
    # regularisation
    p.add_argument("--pll-gamma", type=float, default=0.0, dest="gamma")
    p.add_argument("--pll-shrink", type=float, default=0.0, dest="shrink")
    p.add_argument("--pll-post-gamma", type=float, default=0.05, dest="post_gamma")
    p.add_argument("--pll-post-shrink", type=float, default=0.1, dest="post_shrink")
    p.add_argument("--pll-post-len", type=int, default=16, dest="post_len")
    p.add_argument("--outdir", type=str, default="reports_forecast")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base_minutes = parse_interval_to_minutes(args.interval)
    horizons = parse_horizons(args.horizons, base_minutes)

    # load
    if args.csv:
        df = normalize_ohlcv(pd.read_csv(args.csv))
    else:
        df = load_or_download(args.symbol, args.interval, args.limit, None, outdir)

    # roll
    rows, debug = build_roll_forecast(
        df, args.variants, args.window, horizons, args.horizon_alpha,
        args.qL, args.qS, args.r_trend,
        args.topn, args.min_period_bars,
        args.qA, args.qB, args.qphi, args.qomega, args.qdomega, args.R_meas,
        args.guard_k, args.guard_boost, args.guard_max_misses,
        args.reseed_window, args.reseed_topn,
        args.dw_max, args.d2w_max,
        args.gamma, args.shrink, args.post_gamma, args.post_shrink, args.post_len
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
