#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_predictor.py  (fixed + improved)

- FIX: chybějící import Path (pro ukládání CSV).
- IMPROVE: stabilnější výpočet Jacobiho θ3 pro extrémní q (clamp + adaptivní truncace).
- IMPROVE: korektní projekce z QR prostoru pro predikci: beta_full = solve(R, beta_Q) a pred = B_fut @ beta_full.

Funkce:
- Numerická predikce i směrový signál (LONG/SHORT/FLAT).
- Data z Binance REST API.
- Model: θ3 báze (Jacobi), QR ortonormalizace (volitelně), ridge, projekce do budoucí fáze.
- Volitelný jednoduchý Kalman filtr pro vyhlazení směrového skóre.

Autor: (c) 2025  |  Licence: MIT
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

from pathlib import Path
import numpy as np
import pandas as pd
import requests


# ---------------------------- Theta báze ----------------------------

def theta3(z: np.ndarray, q: float, n_terms: int = 20, tol: float = 1e-12) -> np.ndarray:
    """
    Jacobiho theta funkce θ3(z, q) = 1 + 2 * sum_{n=1..∞} q^{n^2} cos(2 n z)

    Stabilita:
      - q omezíme do (1e-12, 0.5) → pro q>=0.5 by série konvergovala pomalu (nebo numericky hůř),
        doporučujeme q ~ 0.02–0.2 dle specky (sigma≈0.8 → q≈0.081).
      - adaptivní truncace: ukončíme, jakmile q_pow < tol.

    Parametry:
      z: vektor fází
      q: (0,1)
      n_terms: maximální počet členů
      tol: prah pro adaptivní ukončení
    """
    z = np.asarray(z, dtype=np.float64)
    q = float(q)
    # Stabilizační clamp
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1).")
    q = max(1e-12, min(0.5, q))

    out = np.ones_like(z)
    q_pow = q  # q^{1^2}
    n = 1
    while n <= n_terms and q_pow > tol:
        out += 2.0 * q_pow * np.cos(2.0 * n * z)
        # příští člen: q^{(n+1)^2} = q^{n^2} * q^{2n+1}
        q_pow *= q ** (2 * n + 1)
        n += 1
    return out


def build_theta_basis(t_idx: np.ndarray,
                      periods: List[float],
                      q: float,
                      phases: Optional[List[float]] = None,
                      n_terms: int = 20) -> np.ndarray:
    """
    Sestaví matici bází B o tvaru [len(t_idx) x (len(periods) * n_phase + 1)]
    - Sloupec 0: konstanta (bias).
    - Pro každý period P a každou fázi φ vytvoří θ3( ω t + φ, q ), kde ω = 2π / P.
    """
    if phases is None:
        phases = [0.0, np.pi/2]

    cols = [np.ones_like(t_idx, dtype=np.float64)]
    for P in periods:
        omega = 2.0 * np.pi / P
        for phi in phases:
            z = omega * t_idx + phi
            cols.append(theta3(z, q, n_terms=n_terms))
    B = np.vstack(cols).T  # [T x K]
    return B


def orthonormalize_qr(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ QR: B = Q R, kde Q^T Q = I """
    Q, R = np.linalg.qr(B, mode='reduced')
    return Q, R


def ridge_fit(B: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """ Ridge regrese: beta = (B^T B + lam I)^{-1} B^T y """
    BtB = B.T @ B
    K = BtB.shape[0]
    reg = lam * np.eye(K)
    beta = np.linalg.solve(BtB + reg, B.T @ y)
    return beta


# ---------------------------- Kalman (volitelně) ----------------------------

@dataclass
class SimpleKalman:
    R: float = 1e-4  # měřicí šum
    Q: float = 1e-5  # procesní šum
    x: float = 0.0   # stav
    P: float = 1.0   # kovariance

    def update(self, z: float) -> float:
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x


# ---------------------------- Binance data ----------------------------

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

def fetch_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """
    Stáhne OHLCV z Binance veřejného REST API.
    Vrací DataFrame se sloupci: open, high, low, close, volume (index: open_time).
    """
    params = dict(symbol=symbol.upper(), interval=interval, limit=limit)
    r = requests.get(BINANCE_KLINES, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.set_index("open_time")
    return df[["open","high","low","close","volume"]]


# ---------------------------- Predikce ----------------------------

def prepare_series(df: pd.DataFrame, price_col: str = "close", use_log: bool = True) -> np.ndarray:
    """ Vrátí (log-)cenu jako vektor y. """
    arr = df[price_col].values.astype(np.float64)
    return np.log(arr) if use_log else arr


def make_period_grid(minP: float, maxP: float, nP: int) -> List[float]:
    """ Log-rozložená mřížka period P mezi <minP, maxP> (v počtu svíček). """
    return list(np.exp(np.linspace(np.log(minP), np.log(maxP), nP)))


def fit_predict(y: np.ndarray,
                W: int,
                horizon: int,
                q: float,
                periods: List[float],
                lam: float,
                use_qr: bool,
                use_kalman: bool,
                theta_terms: int) -> Tuple[float, float, float]:
    """
    Natrénuje bázi na okně posledních W vzorků a predikuje y_{t+h}.
    Vrací (pred_log_price, last_log_price, directional_score).
    """
    T = len(y)
    if T < W + 2:
        raise ValueError("Málo dat pro zadané W.")

    yw = y[T - W:T]
    t_idx = np.arange(W, dtype=np.float64)

    # Báze v tréninkovém okně
    B = build_theta_basis(t_idx, periods=periods, q=q, n_terms=theta_terms)

    if use_qr:
        Q, R = orthonormalize_qr(B)
        Bfit = Q
        beta_Q = ridge_fit(Bfit, yw, lam=lam)
        # Přepočet do původního prostoru bází: B ≈ Q R ⇒ R * beta_full ≈ beta_Q
        beta_full = np.linalg.solve(R, beta_Q)
    else:
        Bfit = B
        beta_full = ridge_fit(Bfit, yw, lam=lam)

    # Predikční báze v t+h (posun fáze)
    t_future = t_idx[-1] + horizon
    Bfut = build_theta_basis(np.array([t_future], dtype=np.float64),
                             periods=periods, q=q, n_terms=theta_terms)

    pred_log = float((Bfut @ beta_full).ravel()[0])
    last_log = float(y[-1])
    dir_score = pred_log - last_log

    if use_kalman:
        kf = SimpleKalman()
        dir_score = kf.update(dir_score)

    return pred_log, last_log, dir_score


def decide_signal(dir_score: float, fee_bps: float, threshold_sigma: float = 0.0) -> str:
    """ LONG/SHORT/FLAT podle směrového skóre a poplatku v bps. """
    fee_log = 2.0 * (fee_bps / 10000.0)
    if dir_score > fee_log + threshold_sigma:
        return "LONG"
    elif dir_score < -fee_log - threshold_sigma:
        return "SHORT"
    else:
        return "FLAT"


# ---------------------------- CLI ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Theta-basis crypto predictor (numeric + directional)")
    p.add_argument("--symbol", required=True, type=str, help="např. BTCUSDT")
    p.add_argument("--interval", default="1h", type=str, help="Binance interval (1m,5m,15m,1h,4h,1d...)")
    p.add_argument("--window", "--W", dest="W", default=256, type=int, help="délka tréninkového okna")
    p.add_argument("--horizons", default="1,4,24", type=str, help="horizonty v počtu svíček, např. '1,4,24'")
    p.add_argument("--minP", default=24.0, type=float, help="min perioda (počet svíček)")
    p.add_argument("--maxP", default=480.0, type=float, help="max perioda (počet svíček)")
    p.add_argument("--nP", default=8, type=int, help="počet period v mřížce")
    p.add_argument("--sigma", default=0.8, type=float, help="určuje q = exp(- (pi*sigma)^2 ). Specka: sigma≈0.8")
    p.add_argument("--lambda", dest="lam", default=1e-3, type=float, help="ridge regularizace")
    p.add_argument("--no-qr", action="store_true", help="vypne QR ortonormalizaci")
    p.add_argument("--no-kalman", action="store_true", help="vypne jednoduchý Kalman na dir_score")
    p.add_argument("--fee-bps", default=5.0, type=float, help="poplatek v bps pro rozhodování signálu")
    p.add_argument("--use-log", action="store_true", help="použít log-cenu (default ANO).")
    p.add_argument("--theta-terms", default=20, type=int, help="max počet členů série v θ3 (default 20)")
    p.add_argument("--out", default=None, type=str, help="volitelně CSV výstup")
    return p.parse_args()


def main():
    args = parse_args()

    # q ze sigma: q = exp(- (pi * sigma)^2 )
    q = math.exp(- (math.pi * args.sigma)**2)

    # Period grid (v počtu svíček)
    periods = make_period_grid(args.minP, args.maxP, args.nP)

    # Stažení dat
    try:
        df = fetch_klines(args.symbol, args.interval, limit=max(1000, args.W + 10))
    except Exception as e:
        print(f"[ERROR] fetching klines: {e}", file=sys.stderr)
        sys.exit(2)

    y = prepare_series(df, use_log=True)  # držíme log-cenu (robustní pro násobky)

    horizons = [int(h.strip()) for h in str(args.horizons).split(",") if h.strip()]
    rows = []
    for h in horizons:
        pred_log, last_log, dir_score = fit_predict(
            y=y,
            W=args.W,
            horizon=h,
            q=q,
            periods=periods,
            lam=args.lam,
            use_qr=not args.no_qr,
            use_kalman=not args.no_kalman,
            theta_terms=args.theta_terms
        )
        signal = decide_signal(dir_score, fee_bps=args.fee_bps)
        last_price = math.exp(last_log)
        pred_price = math.exp(pred_log)
        rows.append(dict(
            symbol=args.symbol.upper(),
            interval=args.interval,
            window=args.W,
            horizon=h,
            sigma=args.sigma,
            q=q,
            lam=args.lam,
            fee_bps=args.fee_bps,
            last_price=last_price,
            pred_price=pred_price,
            delta=pred_price - last_price,
            dir_score=dir_score,
            signal=signal,
            t_observed=df.index[-1].isoformat()
        ))

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"\nUloženo: {args.out}")


if __name__ == "__main__":
    main()
