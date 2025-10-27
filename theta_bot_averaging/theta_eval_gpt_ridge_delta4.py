#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_eval_gpt_ridge_delta4.py
---------------------------------
Verze bez Ridge regrese — čistá theta báze s projekcí (lineární least-squares).
Slouží pro testování teoretických vlastností theta-transformace
a validaci hypotézy o prediktivní struktuře časových řad.

Autor: David Jaroš (2025)
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from datetime import datetime


# ===============================================================
# === Theta báze ================================================
# ===============================================================

def theta_basis(window, q=0.5, D=0.1):
    """
    Vytvoří jednoduchou reálnou + imaginární bázi založenou na Jacobi theta funkci.
    Zde použita aproximace pomocí Fourierovských harmonických složek.

    Args:
        window (int): délka trénovacího okna
        q (float): parametr (kvazi-perioda)
        D (float): difuzní konstanta – určuje šířku spektra

    Returns:
        np.ndarray: matice [window, 2], sloupce = Re a Im složka
    """
    t = np.linspace(0, 1, window)
    freq = np.pi * q
    theta_re = np.cos(2 * np.pi * freq * t) * np.exp(-D * t)
    theta_im = np.sin(2 * np.pi * freq * t) * np.exp(-D * t)
    basis = np.vstack([theta_re, theta_im]).T
    # ortonormalizace pro stabilitu
    basis /= np.linalg.norm(basis, axis=0)
    return basis


# ===============================================================
# === Hlavní výpočet ============================================
# ===============================================================

def walk_forward_theta(prices, window=512, horizon=32, q=0.5, D=0.1, v=0.0):
    """
    Provede walk-forward test predikce pomocí theta báze.

    Args:
        prices (array): časová řada
        window (int): délka trénovacího okna
        horizon (int): predikční horizont
        q (float): parametr theta funkce
        D (float): difuzní konstanta
        v (float): drift (zatím nevyužit)

    Returns:
        dict: výsledek s metrikou r2_score a sériemi predikcí
    """
    n = len(prices)
    preds, actuals = [], []
    basis = theta_basis(window, q=q, D=D)

    for i in range(window, n - horizon):
        y_train = prices[i - window:i]
        coef, *_ = np.linalg.lstsq(basis, y_train, rcond=None)
        y_pred = np.dot(basis[-1], coef)
        preds.append(y_pred)
        actuals.append(prices[i + horizon - 1])

    if len(preds) < 2:
        return {"r2_score": np.nan, "preds": [], "actuals": []}

    r2 = r2_score(actuals, preds)
    return {"r2_score": float(r2), "preds": preds, "actuals": actuals}


# ===============================================================
# === Exportovaná funkce pro experimenty ========================
# ===============================================================

def run_theta_eval(df, q=0.5, D=0.1, v=0.0, window=512, horizon=32, shuffle=False):
    """
    Exportní funkce — spouští výpočet na libovolném DataFrame.
    Volána z theta_experiments.py.

    Args:
        df (pd.DataFrame): obsahuje sloupec 'close'
        shuffle (bool): zda náhodně promíchat řádky (test pseudonáhodnosti)
    """
    data = df.copy()
    if shuffle:
        data = data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    prices = np.array(data["close"].values, dtype=float)
    return walk_forward_theta(prices, window=window, horizon=horizon, q=q, D=D, v=v)


# ===============================================================
# === CLI rozhraní ==============================================
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="Theta model (bez Ridge)")
    parser.add_argument("--symbols", required=True, help="CSV se symboly nebo daty")
    parser.add_argument("--csv-time-col", default="time")
    parser.add_argument("--csv-close-col", default="close")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--D", type=float, default=0.1)
    parser.add_argument("--v", type=float, default=0.0)
    parser.add_argument("--shuffle", type=int, default=0, help="1 = náhodně promíchat řádky")
    parser.add_argument("--out", required=True, help="výstupní CSV soubor")

    args = parser.parse_args()

    print(f"=== Running {args.symbols} ===")
    df = pd.read_csv(args.symbols)
    df = df.rename(columns={args.csv_close_col: "close", args.csv_time_col: "time"})

    result = run_theta_eval(
        df,
        q=args.q,
        D=args.D,
        v=args.v,
        window=args.window,
        horizon=args.horizon,
        shuffle=bool(args.shuffle)
    )

    # Uložení výsledků
    out_df = pd.DataFrame({
        "time": df["time"].iloc[-len(result["preds"]):].values,
        "pred": result["preds"],
        "actual": result["actuals"]
    })
    out_df.to_csv(args.out, index=False)

    print(f"[DONE] Results saved to {args.out}")
    print(f"R² = {result['r2_score']:.6f}")


if __name__ == "__main__":
    main()

