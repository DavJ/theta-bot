#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_gpt_ridge_delta3.py
==============================

ČISTÁ UBT VARIANTA BEZ RIDGE REGULARIZACE
-----------------------------------------
Tento skript analyzuje časovou řadu (např. ceny BTC) pomocí aproximace
Jacobiho theta funkcí a Fokker–Planckovy drift-difúzní dynamiky,
bez lineární penalizace typu Ridge.

Cílem je zachovat emergentní dynamiku pole Θ(q, τ) = Re[θ3(x,q)] + i·Im[θ2(x,q)],
aniž by došlo k jejímu zkreslení eukleidovskou metrikou.

Autor: Ing. David Jaroš
Verze: delta3 (čistá UBT)
Datum: 2025-10-26
"""

import numpy as np
import pandas as pd
import argparse
from scipy.special import ellipj
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d
import os


# ============================================================
# 🧮 Pomocné funkce
# ============================================================

def theta_basis(N, q=0.5):
    """
    Generuje aproximaci Jacobiho theta báze pomocí eliptických funkcí.
    SciPy >= 1.13 už neobsahuje theta2/theta3, proto použijeme transformaci:
        theta3 ~ cn(u, m)
        theta2 ~ sn(u, m)
    kde m = 1 - q^2
    """
    x = np.linspace(0, 2*np.pi, N)
    m = 1 - q**2
    sn, cn, dn, _ = ellipj(x, m)
    t3 = cn  # aproximace theta3
    t2 = sn  # aproximace theta2
    return t3, t2


def fokker_planck_update(phi, D=0.1, v=0.0, dt=1.0):
    """
    Simuluje evoluci pole podle 1D Fokker–Planckovy rovnice:
        ∂φ/∂t = D ∂²φ/∂x² - v ∂φ/∂x

    - D je difuzní konstanta (rozptyl)
    - v je drift (směrný posun)
    - dt je krok v čase
    """
    grad = np.gradient(phi)
    lapl = np.gradient(grad)
    return phi + dt * (D * lapl - v * grad)


def ema(series, alpha=0.1):
    """Jednoduchý exponenciální klouzavý průměr."""
    return pd.Series(series).ewm(alpha=alpha).mean().values


def normalize(x):
    """Normalizace do intervalu [-1, 1]."""
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1


# ============================================================
# 🧠 Hlavní výpočetní třída
# ============================================================

class ThetaUBTModel:
    """
    Model UBT bez Ridge regularizace.
    Používá Fokker–Planckovskou evoluci pro odhad drifto-difúzní dynamiky
    theta reprezentace časové řady.
    """

    def __init__(self, q=0.5, D=0.1, v=0.0):
        self.q = q
        self.D = D
        self.v = v

    def fit_transform(self, prices):
        """
        Transformuje časovou řadu na theta bázi a nechá ji evolvovat.
        """
        N = len(prices)
        prices_norm = normalize(prices)
        t3, t2 = theta_basis(N, q=self.q)
        phi = prices_norm * t3 + 1j * prices_norm * t2
        # Evoluce podle Fokker–Planckovy rovnice
        phi_next = fokker_planck_update(phi, D=self.D, v=self.v)
        return np.real(phi_next), np.imag(phi_next)


# ============================================================
# 📊 Hlavní běh programu
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Theta UBT (bez Ridge)")
    parser.add_argument("--symbols", type=str, required=True, help="CSV soubor s daty")
    parser.add_argument("--csv-time-col", type=str, default="time")
    parser.add_argument("--csv-close-col", type=str, default="close")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--D", type=float, default=0.1)
    parser.add_argument("--v", type=float, default=0.0)
    parser.add_argument("--ema-alpha", type=float, default=0.0)
    parser.add_argument("--out", type=str, default="results_gpt_ridge_delta/test_v3.csv")
    parser.add_argument("--shuffle", type=int, default=0)
    args = parser.parse_args()

    print(f"=== Running {args.symbols} (UBT pure mode) ===")

    df = pd.read_csv(args.symbols)
    prices = df[args.csv_close_col].values.astype(float)

    if args.shuffle:
        print("[INFO] Shuffling dataset (pseudonáhodná permutace)...")
        np.random.seed(42)
        np.random.shuffle(prices)

    model = ThetaUBTModel(q=args.q, D=args.D, v=args.v)
    re_part, im_part = model.fit_transform(prices)

    mag = np.sqrt(re_part**2 + im_part**2)
    mag_smoothed = gaussian_filter1d(mag, sigma=2)

    if args.ema_alpha > 0:
        mag_smoothed = ema(mag_smoothed, args.ema_alpha)

    df_out = pd.DataFrame({
        "time": df[args.csv_time_col],
        "price": prices,
        "theta_real": re_part,
        "theta_imag": im_part,
        "magnitude": mag_smoothed
    })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"[DONE] Results saved to {args.out}")


# ============================================================
# 🧩 Spuštění
# ============================================================

if __name__ == "__main__":
    main()

