#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Theta Biquaternionic TRUE (BCH-2) — with weighted-QR ortho guards + Kalman/post-scale
-------------------------------------------------------------------------------------
Blocks (2 cols each: cos, sin): time, lin_psi, lin_phi, lin_xi, quad_xx, quad_yy, quad_zz, cross_xy, cross_xz, cross_yz
Selection: block-OMP with weighted-QR refit and guards (coherence μ, condition κ)
Post-proc: optional extra post_scale (shrink) and 1D Kalman smoothing across rolling predictions per horizon.

Outputs stay compatible with your evaluators: pred_raw_h*, pred_thetaBiquatTrue_h*
"""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    import requests  # optional (Binance)
except Exception:
    requests = None

# ---------------- IO helpers ----------------
def parse_interval_to_minutes(interval: str) -> int:
    s = interval.strip().lower()
    num_str = ''.join(ch for ch in s if ch.isdigit())
    unit = ''.join(ch for ch in s if ch.isalpha())
    if not num_str or not unit:
        raise ValueError(f'Invalid interval: {interval}')
    num = int(num_str)
    return {'m':num,'h':60*num,'d':1440*num,'w':10080*num}[unit]

def parse_horizons(tokens: List[str], base_minutes: int) -> List[int]:
    out = []
    for t in tokens:
        t = t.strip().lower()
        if t.isdigit():
            out.append(int(t)); continue
        num_str = ''.join(ch for ch in t if ch.isdigit())
        unit = ''.join(ch for ch in t if ch.isalpha())
        if not num_str or not unit:
            raise ValueError(f'Invalid horizon token: {t}')
        num = int(num_str)
        minutes = {'m':num,'h':60*num,'d':1440*num,'w':10080*num}[unit]
        bars = max(1, round(minutes/base_minutes))
        out.append(bars)
    return sorted(set(out))

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    short2long = {'t':'timestamp','o':'open','h':'high','l':'low','c':'close','v':'volume'}
    if any(c in df.columns for c in short2long):
        df.rename(columns={k:v for k,v in short2long.items() if k in df.columns}, inplace=True)
    alt = {'Timestamp':'timestamp','Time':'timestamp','open_time':'timestamp',
           'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}
    for k,v in alt.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)
    req = ['timestamp','open','high','low','close','volume']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing columns {missing}')
    ts = pd.to_datetime(df['timestamp'], errors='coerce', unit='ms')
    if ts.isna().all():
        ts = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = ts
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['timestamp','close']).sort_values('timestamp').reset_index(drop=True)

def fetch_klines_binance(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    if requests is None:
        raise RuntimeError('requests not available; use --csv')
    url = 'https://api.binance.com/api/v3/klines'
    max_batch = 1000
    remaining = int(limit)
    rows = []
    params = {'symbol': symbol.upper(), 'interval': interval, 'limit': min(max_batch, remaining)}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    batch = r.json()
    if not batch: raise RuntimeError('Empty klines response')
    rows.extend(batch); remaining -= len(batch)
    end_time = int(batch[0][0]) - 1
    while remaining > 0:
        params = {'symbol': symbol.upper(), 'interval': interval, 'limit': min(max_batch, remaining), 'endTime': end_time}
        r = requests.get(url, params=params, timeout=30); r.raise_for_status()
        batch = r.json()
        if not batch: break
        rows.extend(batch); remaining -= len(batch)
        end_time = int(batch[0][0]) - 1
    rows.reverse()
    out = [{
        'timestamp': int(k[0]),
        'open': float(k[1]),
        'high': float(k[2]),
        'low':  float(k[3]),
        'close': float(k[4]),
        'volume': float(k[5]),
    } for k in rows]
    return normalize_ohlcv(pd.DataFrame(out))

def load_or_download(symbol: str, interval: str, limit: int, csv_path: Optional[str], outdir: Path) -> pd.DataFrame:
    if csv_path:
        return normalize_ohlcv(pd.read_csv(csv_path))
    cache = outdir / f'ohlcv_{symbol}_{interval}_{limit}.csv'
    if cache.exists():
        return normalize_ohlcv(pd.read_csv(cache))
    df = fetch_klines_binance(symbol, interval, limit)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df

# ---------------- math helpers ----------------
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

def ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    if len(arr)==0: return arr
    out = np.empty_like(arr, dtype=float)
    s=0.0; a=float(alpha); one=1.0-a
    for i,v in enumerate(arr):
        s = float(v) if i==0 else a*float(v)+one*s
        out[i]=s
    return out

def forecast_raw(window_vals: np.ndarray, horizon: int) -> float:
    return float(window_vals[-1])

# ---------------- biquat components ----------------
def parse_triplet(s: str, default: Tuple[float,float,float]) -> Tuple[float,float,float]:
    try:
        p = [x.strip() for x in s.split(',')]
        if len(p)!=3: return default
        return (float(p[0]), float(p[1]), float(p[2]))
    except Exception:
        return default

def make_biq_components(y: np.ndarray, mode: str,
                        const_triplet: Tuple[float,float,float],
                        scale_triplet: Tuple[float,float,float],
                        ema_triplet: Tuple[int,int,int]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    n = len(y)
    dy = np.diff(y, prepend=y[0])
    absret = np.abs(dy)
    if mode == 'const':
        psi = np.full(n, const_triplet[0], dtype=float)
        phi = np.full(n, const_triplet[1], dtype=float)
        xi  = np.full(n, const_triplet[2], dtype=float)
        return psi, phi, xi
    a1 = 2.0/max(1, ema_triplet[0]+1)
    a2 = 2.0/max(1, ema_triplet[1]+1)
    a3 = 2.0/max(1, ema_triplet[2]+1)
    s1 = scale_triplet[0] * ema(absret, a1)
    s2 = scale_triplet[1] * ema(absret**2, a2)
    s3 = scale_triplet[2] * ema(absret - ema(absret, a1), a3)
    denom = max(1e-12, np.nanmean(np.abs(y)))
    psi = s1/denom; phi = s2/(denom**2); xi = s3/denom
    return psi, phi, xi

def make_weights(y: np.ndarray, w_alpha: float, w_ema: int) -> np.ndarray:
    if w_alpha<=0: return np.ones(len(y), dtype=float)
    dy = np.diff(y, prepend=y[0])
    absret = np.abs(dy) / max(1e-12, np.nanmean(np.abs(y)))
    a = 2.0/max(1, w_ema+1)
    return 1.0 + w_alpha * ema(absret, a)

# ---------------- block factory ----------------
def trig2(arg: np.ndarray) -> np.ndarray:
    c = np.cos(arg); s = np.sin(arg)
    return np.column_stack([c, s])

def make_block(kind: str, omega: float,
               t: np.ndarray, psi: np.ndarray, phi: np.ndarray, xi: np.ndarray) -> np.ndarray:
    if kind == 'time':      return trig2(omega * t)
    if kind == 'lin_psi':   return trig2(omega * psi)
    if kind == 'lin_phi':   return trig2(omega * phi)
    if kind == 'lin_xi':    return trig2(omega * xi)
    if kind == 'quad_xx':   return trig2(omega * (psi**2))
    if kind == 'quad_yy':   return trig2(omega * (phi**2))
    if kind == 'quad_zz':   return trig2(omega * (xi**2))
    if kind == 'cross_xy':  return trig2(omega * (psi*phi))
    if kind == 'cross_xz':  return trig2(omega * (psi*xi))
    if kind == 'cross_yz':  return trig2(omega * (phi*xi))
    raise ValueError(f'Unknown block kind: {kind}')

BLOCK_KINDS = ['time','lin_psi','lin_phi','lin_xi','quad_xx','quad_yy','quad_zz','cross_xy','cross_xz','cross_yz']

def weighted_qr(C: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float=0.0):
    W = np.sqrt(w)[:,None]
    Cw = C * W; yw = y[:,None] * W
    if lam>0.0 and C.shape[1]>0:
        d = C.shape[1]
        Cw = np.vstack([Cw, np.sqrt(lam)*np.eye(d)])
        yw = np.vstack([yw, np.zeros((d,1))])
    if Cw.shape[1]==0:
        # no regressors
        resid = yw
        rss = float((resid.T @ resid).ravel()[0])
        return np.zeros(0), rss
    Q,R = np.linalg.qr(Cw, mode='reduced')
    beta = np.linalg.solve(R, Q.T @ yw)
    resid = yw - Cw @ beta
    rss = float((resid.T @ resid).ravel()[0])
    return beta.ravel(), rss

def gram_and_metrics(C: np.ndarray, w: np.ndarray):
    W = np.sqrt(w)[:,None]
    Cw = C * W
    if Cw.shape[1]==0:
        return np.zeros((0,0)), 0.0, 1.0
    G = Cw.T @ Cw
    diag = np.sqrt(np.maximum(1e-18, np.diag(G)))
    if G.shape[0] <= 1:
        mu = 0.0
    else:
        Gnorm = G / (diag[:,None]*diag[None,:] + 1e-18)
        mu = float(np.max(np.abs(Gnorm - np.eye(G.shape[0]))))
    s = np.linalg.svd(G, compute_uv=False)
    kappa = float((s[0]/max(s[-1],1e-18)) if s.size else 1.0)
    return G, mu, kappa

# ---------------- OMP with guards ----------------
def omp_biquat(y: np.ndarray, t: np.ndarray, psi: np.ndarray, phi: np.ndarray, xi: np.ndarray,
               omega_grid: np.ndarray,
               w: np.ndarray, lam: float, topn: int, scan_every:int,
               max_coherence: float, max_cond: float, log_steps: bool):
    a,b = _lin_trend(y)
    r = y - (a*t + b)
    n = len(y)
    active = []
    C_sel = np.zeros((n,0))
    _, rss_best = weighted_qr(C_sel, r, w, lam=0.0)
    steps_log = []
    for it in range(max(1, topn)):
        best = None; best_gain = 0.0; best_mu=None; best_kappa=None
        for idx, om in enumerate(omega_grid):
            if idx % max(1,scan_every) != 0:
                continue
            for kind in BLOCK_KINDS:
                Cb = make_block(kind, om, t, psi, phi, xi)  # n x 2
                C_try = np.hstack([C_sel, Cb]) if C_sel.size else Cb
                G, mu, kappa = gram_and_metrics(C_try, w)
                if mu > max_coherence or kappa > max_cond:
                    continue
                _, rss_try = weighted_qr(C_try, r, w, lam=lam if C_sel.size else 0.0)
                gain = rss_best - rss_try
                if gain > best_gain:
                    best_gain = gain; best = (kind, float(om), C_try, rss_try); best_mu=mu; best_kappa=kappa
        if best is None or best_gain <= 0.0:
            break
        kind, om, C_sel, rss_best = best
        active.append((kind, om))
        if log_steps:
            steps_log.append({'iter': it+1, 'kind': kind, 'omega': om,
                              'gain': float(best_gain), 'coherence_mu': float(best_mu), 'cond_kappa': float(best_kappa)})
    if not active:
        return active, np.zeros(0), a, b, steps_log
    C_final = np.hstack([make_block(kind, om, t, psi, phi, xi) for (kind,om) in active])
    beta, _ = weighted_qr(C_final, r, w, lam=lam)
    return active, beta, a, b, steps_log

def synth_from_blocks(active, beta, t_val, psi_val, phi_val, xi_val) -> float:
    assert len(beta) == 2*len(active)
    total = 0.0
    for k,(kind,om) in enumerate(active):
        if kind == 'time':      arg = om * t_val
        elif kind == 'lin_psi': arg = om * psi_val
        elif kind == 'lin_phi': arg = om * phi_val
        elif kind == 'lin_xi':  arg = om * xi_val
        elif kind == 'quad_xx': arg = om * (psi_val**2)
        elif kind == 'quad_yy': arg = om * (phi_val**2)
        elif kind == 'quad_zz': arg = om * (xi_val**2)
        elif kind == 'cross_xy':arg = om * (psi_val*phi_val)
        elif kind == 'cross_xz':arg = om * (psi_val*xi_val)
        elif kind == 'cross_yz':arg = om * (phi_val*xi_val)
        else: continue
        c = math.cos(arg); s = math.sin(arg)
        a = beta[2*k]; b = beta[2*k+1]
        total += a*c + b*s
    return float(total)

def forecast_theta_biquat_true(y: np.ndarray, horizon: int,
                               zero_pad: int, min_period_bars: int, lam: float,
                               biq_mode: str, biq_const, biq_scale, biq_ema,
                               w_alpha: float, w_ema: int,
                               topn: int, scan_every: int,
                               max_coherence: float, max_cond: float,
                               damping: float, shrink: float,
                               log_steps: bool):
    y = y.astype(float)
    n = len(y); t = np.arange(n, dtype=float)
    psi,phi,xi = make_biq_components(y, biq_mode, biq_const, biq_scale, biq_ema)
    w = make_weights(y, w_alpha, w_ema)
    N = int(n*max(1,zero_pad))
    freqs = np.fft.rfftfreq(N, d=1.0)
    mask = freqs>0
    if min_period_bars>0:
        mask &= (1.0/np.maximum(freqs,1e-12)) >= min_period_bars
    freqs = freqs[mask]
    if freqs.size==0: 
        return float(y[-1]), {'active':[], 'steps':[]}
    omega_grid = 2.0*np.pi*freqs
    active, beta, a, b, steps = omp_biquat(
        y, t, psi, phi, xi, omega_grid, w, lam=lam, topn=topn, scan_every=scan_every,
        max_coherence=max_coherence, max_cond=max_cond, log_steps=log_steps
    )
    t_f = (n-1) + horizon
    psi_f, phi_f, xi_f = psi[-1], phi[-1], xi[-1]
    synth = synth_from_blocks(active, beta, t_f, psi_f, phi_f, xi_f)
    if shrink>0.0: synth *= (1.0 - shrink)
    if damping>0.0: synth *= float(np.exp(-damping*horizon))
    y_hat = synth + (a*(n-1+horizon)+b)
    debug = {'active': [{'kind':k, 'omega':om} for (k,om) in active], 'steps': steps}
    return float(y_hat), debug

# ---------------- Rolling runner ----------------
def build_roll_forecast(df: pd.DataFrame, variants: List[str], window: int, horizons: List[int],
                        horizon_alpha: float,
                        biq_zero_pad: int, biq_min_period_bars: int, biq_lam: float,
                        biq_mode: str, biq_const, biq_scale, biq_ema,
                        biq_w_alpha: float, biq_w_ema: int,
                        biq_topn: int, biq_scan_every: int,
                        biq_max_coherence: float, biq_max_cond: float,
                        biq_damping: float, biq_shrink: float,
                        log_steps: bool,
                        post_kalman: bool=False,
                        post_kalman_r_mult: float=1.0,
                        post_kalman_q_mult: float=0.10,
                        post_scale: float=1.0):
    ts = pd.to_datetime(df['timestamp'])
    closes = df['close'].astype(float).to_numpy()
    n = len(closes)
    start = window
    end = n - max(horizons)
    if end <= start:
        raise RuntimeError(f'Nedostatek dat: n={n}, window={window}, horizons={horizons}')
    rows = []
    logs = []
    # --- Kalman state per horizon (random-walk model) ---
    kf_state = {H: {'m': None, 'P': None} for H in horizons}

    def kf_update(H, meas, local_var):
        st = kf_state[H]
        # Initialize if needed
        if st['m'] is None:
            st['m'] = float(meas)
            st['P'] = float(local_var) if local_var>0 else 1.0
            return st['m']
        # Model: x_t = x_{t-1} + w,  y_t = x_t + v
        R = post_kalman_r_mult * max(local_var, 1e-12)
        Q = post_kalman_q_mult * R
        # Predict
        m_pred = st['m']
        P_pred = st['P'] + Q
        # Update
        K = P_pred / (P_pred + R)
        m = m_pred + K * (meas - m_pred)
        P = (1.0 - K) * P_pred
        st['m'], st['P'] = float(m), float(P)
        return st['m']

    for t_idx in range(start, end):
        w = closes[t_idx-window:t_idx]
        row = {'timestamp': ts.iloc[t_idx].isoformat(), 'close': float(closes[t_idx])}
        log_row = {'timestamp': ts.iloc[t_idx].isoformat(), 'entries': {}}
        for H in horizons:
            if 'raw' in variants:
                v = forecast_raw(w, H)
                row[f'pred_raw_h{H}'] = float(horizon_alpha*v + (1.0-horizon_alpha)*row['close'])
            if 'thetaBiquatTrue' in variants:
                v, dbg = forecast_theta_biquat_true(
                    w, H,
                    zero_pad=biq_zero_pad, min_period_bars=biq_min_period_bars, lam=biq_lam,
                    biq_mode=biq_mode, biq_const=biq_const, biq_scale=biq_scale, biq_ema=biq_ema,
                    w_alpha=biq_w_alpha, w_ema=biq_w_ema,
                    topn=biq_topn, scan_every=biq_scan_every,
                    max_coherence=biq_max_coherence, max_cond=biq_max_cond,
                    damping=biq_damping, shrink=biq_shrink,
                    log_steps=log_steps
                )
                key = f'pred_thetaBiquatTrue_h{H}'
                base_pred = float(horizon_alpha*v + (1.0-horizon_alpha)*row['close'])
                # Extra post scaling (relative to current close)
                if post_scale != 1.0:
                    base_pred = float(row['close'] + post_scale * (base_pred - row['close']))
                # Optional Kalman smoothing across rolling predictions
                if post_kalman:
                    # local variance proxy: variance of last window diffs
                    diffs = np.diff(w).astype(float)
                    local_var = float(np.var(diffs)) if diffs.size>0 else 0.0
                    base_pred = float(kf_update(H, base_pred, local_var))
                row[key] = base_pred
                log_row['entries'][key] = dbg
        rows.append(row)
        if log_steps:
            logs.append(log_row)
    debug = {'params':{
        'window':window,'horizons':horizons,'horizon_alpha':horizon_alpha,
        'thetaBiquatTrue':{
            'zero_pad':biq_zero_pad,'min_period_bars':biq_min_period_bars,'lam':biq_lam,
            'mode':biq_mode,'const':biq_const,'scale':biq_scale,'ema':biq_ema,
            'w_alpha':biq_w_alpha,'w_ema':biq_w_ema,
            'topn':biq_topn,'scan_every':biq_scan_every,
            'max_coherence':biq_max_coherence,'max_cond':biq_max_cond,
            'damping':biq_damping,'shrink':biq_shrink,
            'log_steps': log_steps},
        'post': {'post_kalman': post_kalman, 'post_kalman_r_mult': post_kalman_r_mult,
                 'post_kalman_q_mult': post_kalman_q_mult, 'post_scale': post_scale}},
        'n_rows': len(rows),
        'logs': logs if log_steps else None}
    return rows, debug

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Theta Biquaternionic TRUE (BCH-2) with ortho guards + post Kalman/scale')
    p.add_argument('--symbol', required=True)
    p.add_argument('--interval', required=True)
    p.add_argument('--limit', type=int, default=10000)
    p.add_argument('--csv', type=str, default=None, help='Optional CSV with OHLCV; otherwise downloads from Binance')
    p.add_argument('--variants', nargs='+', default=['raw','thetaBiquatTrue'])
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--horizons', nargs='+', default=['1h'])
    p.add_argument('--horizon-alpha', type=float, default=1.0, dest='horizon_alpha')
    # biquat true params
    p.add_argument('--biq-zero-pad', type=int, default=4, dest='biq_zero_pad')
    p.add_argument('--biq-min-period-bars', type=int, default=24, dest='biq_min_period_bars')
    p.add_argument('--biq-lam', type=float, default=1e-5, dest='biq_lam')
    p.add_argument('--biq-mode', type=str, default='ema3', choices=['const','ema3'])
    p.add_argument('--biq-const', type=str, default='0,0,0')
    p.add_argument('--biq-scale', type=str, default='1.0,0.5,0.5')
    p.add_argument('--biq-ema', type=str, default='32,64,32')
    p.add_argument('--biq-w-alpha', type=float, default=0.5, dest='biq_w_alpha')
    p.add_argument('--biq-w-ema', type=int, default=32, dest='biq_w_ema')
    p.add_argument('--biq-topn', type=int, default=2, dest='biq_topn')
    p.add_argument('--biq-scan-every', type=int, default=8, dest='biq_scan_every')
    p.add_argument('--biq-max-coherence', type=float, default=0.25, dest='biq_max_coherence')
    p.add_argument('--biq-max-cond', type=float, default=1e4, dest='biq_max_cond')
    p.add_argument('--biq-damping', type=float, default=0.0, dest='biq_damping')
    p.add_argument('--biq-shrink', type=float, default=0.0, dest='biq_shrink')
    p.add_argument('--log-steps', action='store_true', help='Log active blocks per step into JSON')
    # --- post-processing (stabilizace predikce) ---
    p.add_argument('--post-kalman', action='store_true', help='Apply simple RW Kalman smoothing across rolling predictions per horizon')
    p.add_argument('--post-kalman-r-mult', type=float, default=1.0, dest='post_kalman_r_mult',
                   help='Measurement noise multiplier R relative to local variance')
    p.add_argument('--post-kalman-q-mult', type=float, default=0.10, dest='post_kalman_q_mult',
                   help='Process noise multiplier Q relative to R (Q = q_mult * R)')
    p.add_argument('--post-scale', type=float, default=1.0, dest='post_scale',
                   help='Extra shrink on deviation from close: y = close + post_scale*(y-close)')
    p.add_argument('--outdir', type=str, default='reports_forecast')
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
    def parse_triplet(s: str, default):
        try:
            p = [x.strip() for x in s.split(',')]
            if len(p)!=3: return default
            return (float(p[0]), float(p[1]), float(p[2]))
        except Exception:
            return default
    const_triplet = tuple(float(x) for x in parse_triplet(args.biq_const, (0.0,0.0,0.0)))
    scale_triplet = tuple(float(x) for x in parse_triplet(args.biq_scale, (1.0,0.5,0.5)))
    ema_triplet   = tuple(int(x)   for x in parse_triplet(args.biq_ema, (32,64,32)))
    rows, debug = build_roll_forecast(
        df, args.variants, args.window, horizons, args.horizon_alpha,
        args.biq_zero_pad, args.biq_min_period_bars, args.biq_lam,
        args.biq_mode, const_triplet, scale_triplet, ema_triplet,
        args.biq_w_alpha, args.biq_w_ema,
        args.biq_topn, args.biq_scan_every,
        args.biq_max_coherence, args.biq_max_cond,
        args.biq_damping, args.biq_shrink,
        args.log_steps,
        post_kalman=args.post_kalman,
        post_kalman_r_mult=args.post_kalman_r_mult,
        post_kalman_q_mult=args.post_kalman_q_mult,
        post_scale=args.post_scale
    )
    base = f"{args.symbol}_{args.interval}_win{args.window}_h{'-'.join(map(str,horizons))}_pure{int(args.horizon_alpha==1.0)}"
    csv_path = outdir / f"forecast_{base}.csv"
    log_path = outdir / f"forecast_{base}.json"
    if rows:
        cols = ['timestamp','close'] + sorted([k for k in rows[0].keys() if k.startswith('pred_')])
        pd.DataFrame(rows)[cols].to_csv(csv_path, index=False)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'debug': debug}, f, ensure_ascii=False, indent=2)
    print(f"Saved: {csv_path}")
    print(f"Saved: {log_path}")

if __name__ == '__main__':
    main()
