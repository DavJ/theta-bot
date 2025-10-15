#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Runner with pure-extrapolation FFT (no Kalman/PLL): "fftRefined"

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

# ---------- simple baselines ----------
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

# ---------- refined FFT (pure extrapolation) ----------
def _hann(n: int) -> np.ndarray:
    if n < 1: return np.ones(0)
    i = np.arange(n, dtype=float)
    return 0.5 - 0.5*np.cos(2.0*np.pi*i/(n-1))

def _quadratic_peak_refine(mag: np.ndarray, k: int):
    # Parabolic interpolation around peak k (in log domain for symmetry)
    if k <= 0 or k >= len(mag)-1:
        return 0.0
    y0, y1, y2 = np.log(mag[k-1]+1e-12), np.log(mag[k]+1e-12), np.log(mag[k+1]+1e-12)
    denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5*(y0 - y2)/denom  # offset in [-0.5,0.5] ideally
    return float(np.clip(delta, -0.5, 0.5))

def _peak_indices(mag: np.ndarray, min_separation: int = 1, topn: int = 1):
    # Simple local maxima search
    idxs = []
    for k in range(1, len(mag)-1):
        if mag[k] > mag[k-1] and mag[k] >= mag[k+1]:
            idxs.append(k)
    # sort by magnitude
    idxs = sorted(idxs, key=lambda k: mag[k], reverse=True)
    # enforce separation
    selected = []
    for k in idxs:
        if all(abs(k - s) >= min_separation for s in selected):
            selected.append(k)
        if len(selected) >= topn:
            break
    return selected

def forecast_fft_refined(window_vals: np.ndarray, horizon: int, topn: int = 1,
                         zero_pad: int = 4, min_period_bars: int = 24,
                         damping: float = 0.0, shrink: float = 0.0) -> float:
    y = window_vals.astype(float)
    n = len(y)
    a,b = _lin_trend(y)
    trend = a*np.arange(n)+b
    r = y - trend
    # windowing + zero padding
    w = _hann(n)
    r_w = r * w
    N = int(n * max(1, zero_pad))
    Y = np.fft.rfft(r_w, n=N)
    freqs = np.fft.rfftfreq(N, d=1.0)
    mag = np.abs(Y)
    # ignore DC
    if len(mag) > 0:
        mag[0] = 0.0
    # drop too fast frequencies
    if min_period_bars > 0:
        mask = np.ones_like(freqs, dtype=bool)
        mask[freqs > 0] = (1.0 / np.maximum(freqs[freqs>0], 1e-12)) >= min_period_bars
        mag = mag * mask
    # find peaks
    sep = max(1, int(N / (4*n)))  # coarse separation
    peaks = _peak_indices(mag, min_separation=sep, topn=topn)
    if not peaks:
        return float(y[-1])
    # build refined sinus sum
    t = n + horizon
    synth = 0.0
    for k in peaks:
        # refine frequency
        delta = _quadratic_peak_refine(mag, k)
        k_ref = k + delta
        f_ref = k_ref / N
        # amplitude and phase from complex spectrum at refined bin (nearest)
        # For phase, take the complex value at integer k (good enough after windowing/zero-pad)
        Ak = (2.0 * np.abs(Y[k])) / np.sum(w)  # scale by window energy
        phi = np.angle(Y[k])
        synth += Ak * np.cos(2.0*np.pi*f_ref*t + phi)
    # shrink/damp
    if shrink > 0.0:
        synth *= (1.0 - shrink)
    if damping > 0.0:
        synth *= float(np.exp(-damping * horizon))
    y_hat = synth + (a*(n+horizon)+b)
    return float(y_hat)

# ---------- rolling engine ----------
def build_roll_forecast(df: pd.DataFrame, variants: List[str], window: int, horizons: List[int],
                        horizon_alpha: float,
                        fft_topn: int,
                        fft_refined_topn: int, fft_refined_zero_pad: int,
                        fft_refined_min_period_bars: int, fft_refined_damping: float, fft_refined_shrink: float
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
            if "fftRefined" in variants:
                v = forecast_fft_refined(w, H, topn=fft_refined_topn,
                                         zero_pad=fft_refined_zero_pad,
                                         min_period_bars=fft_refined_min_period_bars,
                                         damping=fft_refined_damping,
                                         shrink=fft_refined_shrink)
                row[f"pred_fftRefined_h{H}"] = float(horizon_alpha*v + (1.0-horizon_alpha)*row["close"])
        rows.append(row)
    debug = {"params":{
        "window":window,"horizons":horizons,"horizon_alpha":horizon_alpha,
        "fft_topn":fft_topn,
        "fft_refined":{"topn":fft_refined_topn,"zero_pad":fft_refined_zero_pad,
                       "min_period_bars":fft_refined_min_period_bars,
                       "damping":fft_refined_damping,"shrink":fft_refined_shrink}
    },"n_rows": len(rows)}
    return rows, debug

# ---------- data loading wrappers ----------
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Theta rolling forecast with pure FFT extrapolation (fftRefined)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--variants", nargs="+", default=["raw","fft","fftRefined"])
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--horizons", nargs="+", default=["1h"])
    p.add_argument("--horizon-alpha", type=float, default=1.0, dest="horizon_alpha")
    p.add_argument("--fft-topn", type=int, default=8, dest="fft_topn")
    p.add_argument("--fft-refined-topn", type=int, default=1, dest="fft_refined_topn")
    p.add_argument("--fft-refined-zero-pad", type=int, default=4, dest="fft_refined_zero_pad")
    p.add_argument("--fft-refined-min-period-bars", type=int, default=24, dest="fft_refined_min_period_bars")
    p.add_argument("--fft-refined-damping", type=float, default=0.0, dest="fft_refined_damping")
    p.add_argument("--fft-refined-shrink", type=float, default=0.0, dest="fft_refined_shrink")
    p.add_argument("--outdir", type=str, default="reports_forecast")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base_minutes = parse_interval_to_minutes(args.interval)
    horizons = parse_horizons(args.horizons, base_minutes)
    df = load_or_download(args.symbol, args.interval, args.limit, args.csv, outdir)
    rows, debug = build_roll_forecast(
        df, args.variants, args.window, horizons, args.horizon_alpha,
        args.fft_topn,
        args.fft_refined_topn, args.fft_refined_zero_pad,
        args.fft_refined_min_period_bars, args.fft_refined_damping, args.fft_refined_shrink
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
