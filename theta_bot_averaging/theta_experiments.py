#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theta_experiments.py
-----------------------------------
Experimentální wrapper pro testování invariance theta báze (CCT/UBT-Δ4).

Provede tři experimenty:
  1. Random Phase Test
  2. Permutation Batch Test
  3. Synthetic Noise Test

Výsledky ukládá do složky ./experiments_delta4/
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Importuj hlavní výpočetní funkci z delta3
from theta_eval_gpt_ridge_delta4 import run_theta_eval

# -----------------------------------------------------------
# Pomocné funkce
# -----------------------------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def add_random_phase(series):
    """Přidá náhodnou fázovou rotaci e^{iφ} ke každému vzorku."""
    phi = np.random.uniform(0, 2 * np.pi)
    return np.real(series * np.exp(1j * phi))

def load_data(symbol_file, time_col, close_col):
    df = pd.read_csv(symbol_file)
    df = df[[time_col, close_col]].dropna()
    df.rename(columns={time_col: "time", close_col: "close"}, inplace=True)
    return df

def summary_stats(results):
    """Vrátí přehled základních statistik z výsledků testů."""
    arr = np.array(results)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

# -----------------------------------------------------------
# Test 1: Random Phase
# -----------------------------------------------------------

def random_phase_test(args, df):
    print("\n🌀 Spouštím Random Phase Test...")
    outdir = os.path.join(args.outdir, "phase_test")
    ensure_dir(outdir)

    results = []
    for i in tqdm(range(args.repeats), desc="Phase rotations"):
        df_phase = df.copy()
        df_phase["close"] = add_random_phase(df_phase["close"])
        res = run_theta_eval(df_phase, args.q, args.D, args.v, args.window, args.horizon)
        results.append(res["r2_score"])
        pd.DataFrame({"r2": [res["r2_score"]]}).to_csv(f"{outdir}/phase_{i:03d}.csv", index=False)

    stats = summary_stats(results)
    json.dump(stats, open(f"{outdir}/summary.json", "w"), indent=2)
    print("✅ Random Phase hotovo:", stats)
    return stats

# -----------------------------------------------------------
# Test 2: Permutation Batch
# -----------------------------------------------------------

def permutation_test(args, df):
    print("\n🔀 Spouštím Permutation Batch Test...")
    outdir = os.path.join(args.outdir, "perm_test")
    ensure_dir(outdir)

    results = []
    for i in tqdm(range(args.repeats), desc="Permutations"):
        df_perm = df.copy()
        df_perm["close"] = np.random.permutation(df_perm["close"].values)
        res = run_theta_eval(df_perm, args.q, args.D, args.v, args.window, args.horizon)
        results.append(res["r2_score"])
        pd.DataFrame({"r2": [res["r2_score"]]}).to_csv(f"{outdir}/perm_{i:03d}.csv", index=False)

    stats = summary_stats(results)
    json.dump(stats, open(f"{outdir}/summary.json", "w"), indent=2)
    print("✅ Permutation hotovo:", stats)
    return stats

# -----------------------------------------------------------
# Test 3: Synthetic Noise
# -----------------------------------------------------------

def noise_test(args, df):
    print("\n📉 Spouštím Synthetic Noise Test...")
    outdir = os.path.join(args.outdir, "noise_test")
    ensure_dir(outdir)

    mu, sigma = df["close"].mean(), df["close"].std()
    results = []
    for i in tqdm(range(args.repeats), desc="Noise runs"):
        df_noise = df.copy()
        df_noise["close"] = np.random.normal(mu, sigma, len(df))
        res = run_theta_eval(df_noise, args.q, args.D, args.v, args.window, args.horizon)
        results.append(res["r2_score"])
        pd.DataFrame({"r2": [res["r2_score"]]}).to_csv(f"{outdir}/noise_{i:03d}.csv", index=False)

    stats = summary_stats(results)
    json.dump(stats, open(f"{outdir}/summary.json", "w"), indent=2)
    print("✅ Noise hotovo:", stats)
    return stats

# -----------------------------------------------------------
# Main orchestrátor
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Theta experimentální sada Δ4")
    parser.add_argument("--symbols", required=True, help="CSV s časovou řadou")
    parser.add_argument("--csv-time-col", default="time", help="Název sloupce s časem")
    parser.add_argument("--csv-close-col", default="close", help="Název sloupce s cenou")
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--D", type=float, default=0.1)
    parser.add_argument("--v", type=float, default=0.0)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--mode", choices=["all", "phase", "perm", "noise"], default="all")
    parser.add_argument("--outdir", default="experiments_delta4/")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    df = load_data(args.symbols, args.csv_time_col, args.csv_close_col)

    summary = {"timestamp": datetime.utcnow().isoformat()}
    if args.mode in ["all", "phase"]:
        summary["phase"] = random_phase_test(args, df)
    if args.mode in ["all", "perm"]:
        summary["perm"] = permutation_test(args, df)
    if args.mode in ["all", "noise"]:
        summary["noise"] = noise_test(args, df)

    json.dump(summary, open(os.path.join(args.outdir, "summary_all.json"), "w"), indent=2)
    print("\n🧭 Kompletní výsledky uložené do:", args.outdir)

# -----------------------------------------------------------

if __name__ == "__main__":
    main()

