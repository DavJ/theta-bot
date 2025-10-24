#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
from mpmath import jtheta, mp

# Nastavení mpmath přesnosti
mp.dps = 50 # Nebo jiná vhodná přesnost

# -----------------------------
# Helpery (stejné jako předtím)
# -----------------------------

def _read_price_csv(path: str) -> pd.DataFrame:
    """Načte CSV a najde sloupce 'time' a 'close'."""
    try: df = pd.read_csv(path)
    except Exception as e: raise SystemExit(f"ERROR: Failed to read CSV '{path}': {e}")

    tcol = None
    time_candidates = ["time", "timestamp", "date", "datetime", "Time", "Timestamp", "open_time"]
    for c in time_candidates:
        if c in df.columns: tcol = c; break
    if tcol is None: df["time"] = np.arange(len(df)); tcol = "time"

    ccol = None
    close_candidates = ["close", "Close", "price", "Price", "close_price"]
    open_candidates = ["open", "Open", "open_price"]
    for c in close_candidates:
        if c in df.columns: ccol = c; break
    if ccol is None:
        for c_open in open_candidates:
             if c_open in df.columns: ccol = c_open; break
        if ccol is None: raise ValueError(f"'{path}': Nenašel jsem sloupec s cenou. Columns: {list(df.columns)}")

    out = df[[tcol, ccol]].copy(); out.columns = ["time", "close"]
    try:
        time_parsed = pd.to_datetime(out['time'], utc=True, errors='coerce')
        if time_parsed.isnull().any() and pd.api.types.is_numeric_dtype(df[tcol]):
             time_parsed = pd.to_datetime(df[tcol], unit='ms', utc=True, errors='coerce')
        if time_parsed.isnull().any(): out['time'] = pd.to_datetime(np.arange(len(out)), unit='s', utc=True)
        else: out['time'] = time_parsed
    except Exception as e: out['time'] = pd.to_datetime(np.arange(len(out)), unit='s', utc=True)
    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    out = out.dropna(subset=['time', 'close']).sort_values('time').reset_index(drop=True)
    out = out.drop_duplicates(subset=['time'], keep='first').reset_index(drop=True)
    if len(out) == 0: raise ValueError(f"No valid data in '{path}'.")
    return out

def ridge(X, y, lam): # Zkopírováno z fixed_leak
    """Řeší Ridge regresi."""
    if X.ndim != 2 or y.ndim != 1:
         if y.ndim == 2 and y.shape[1] == 1: y = y.ravel()
         else: raise ValueError(f"Invalid dimensions for ridge: X={X.shape}, y={y.shape}")
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0: return np.zeros(n_features)
    if n_samples < n_features: print(f"Warning: Underdetermined system ({n_samples}<{n_features}).")
    XT = X.T; A = XT @ X
    if n_features > 0: A += lam * np.identity(n_features)
    b = XT @ y
    try: beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        try: beta = np.linalg.lstsq(A, b, rcond=None)[0]
        except Exception: beta = np.zeros(n_features)
    return beta.reshape(-1)

def _corr(a: np.ndarray, b: np.ndarray) -> float: # Zkopírováno z fixed_leak
    """Bezpečný výpočet korelace."""
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    mask = ~np.isnan(a) & ~np.isnan(b)
    a_clean, b_clean = a[mask], b[mask]
    n_clean = len(a_clean)
    if n_clean < 2: return float("nan")
    var_a = np.var(a_clean); var_b = np.var(b_clean)
    if var_a < 1e-12 or var_b < 1e-12: return 0.0
    try:
        corr_matrix = np.corrcoef(a_clean, b_clean)
        if isinstance(corr_matrix, np.ndarray) and corr_matrix.shape == (2, 2):
            return float(np.clip(corr_matrix[0, 1], -1.0, 1.0))
        else: return float("nan")
    except Exception: return float("nan")

def _safe_stem_key(path: str) -> str: # Zkopírováno z fixed_leak
    """Vytvoří bezpečný klíč ze jména souboru."""
    stem = Path(path).stem
    safe_key = "".join(c if c.isalnum() else '_' for c in stem).upper()
    max_len = 50
    return safe_key[:max_len] + '_TRUNC' if len(safe_key) > max_len else safe_key

@dataclass
class EvalResult: # Zkopírováno z fixed_leak
    """Struktura pro ukládání výsledků metrik."""
    hit_rate_pred: float
    hit_rate_hold: float
    corr_pred_true: float # Korelace DELTA
    mae_delta: float      # MAE DELTA
    mae_return: float
    count: int
    corr_price: float = float('nan')
    anti_hit_rate: float = float('nan')
    zero_rate: float = float('nan')
    mean_delta_prod: float = float('nan')

def metrics_with_diag(last_prices, pred_delta, true_delta): # Přidáno z fixed_leak
    """Vypočítá metriky výkonu VČETNĚ diagnostiky."""
    pred_delta = np.asarray(pred_delta).ravel()
    true_delta = np.asarray(true_delta).ravel()
    last_prices = np.asarray(last_prices).ravel()
    valid_mask = ~np.isnan(pred_delta) & ~np.isnan(true_delta) & ~np.isnan(last_prices)
    count = valid_mask.sum()

    # Inicializace diagnostických metrik na NaN
    corr_delta, corr_price, hit, anti, zeros, sum_check, mean_prod = [np.nan] * 7

    if count >= 2: # Potřebujeme alespoň 2 body pro korelaci a smysluplné průměry
        pd_clean = pred_delta[valid_mask]
        td_clean = true_delta[valid_mask]
        lp_clean = last_prices[valid_mask]
        pp_clean = lp_clean + pd_clean
        fp_clean = lp_clean + td_clean

        corr_delta = _corr(pd_clean, td_clean)
        corr_price = _corr(pp_clean, fp_clean)
        sign_pred = np.sign(pd_clean)
        sign_true = np.sign(td_clean)
        correct = (sign_pred != 0) & (sign_true != 0) & (sign_pred == sign_true)
        opposite = (sign_pred != 0) & (sign_true != 0) & (sign_pred == -sign_true)
        has_zero = (sign_pred == 0) | (sign_true == 0)
        total_valid_compare = len(pd_clean)
        hit = np.sum(correct) / total_valid_compare
        anti = np.sum(opposite) / total_valid_compare
        zeros = np.sum(has_zero) / total_valid_compare
        sum_check = hit + anti + zeros
        mean_prod = np.mean(pd_clean * td_clean)

    print("\n--- DEBUG DIAGNOSTICS ---")
    print(f"[DBG] corr(Δ,Δ)         = {corr_delta:.3f}")
    print(f"[DBG] corr(price,price) = {corr_price:.3f}")
    print(f"[DBG] hit rate        = {hit:.3f}")
    print(f"[DBG] anti-hit rate   = {anti:.3f}")
    print(f"[DBG] zero rate       = {zeros:.3f}")
    print(f"[DBG] sum check (h+a+z) = {sum_check:.3f}")
    print(f"[DBG] mean(Δ_pred*Δ_true) = {mean_prod:.4g}")
    print("-------------------------\n")

    # Standardní metriky (pokud count >= 1)
    hit_pred_std, hit_hold_std, mae_p_std, mae_r_std = [np.nan] * 4
    if count >= 1:
        pd_clean = pred_delta[valid_mask]
        td_clean = true_delta[valid_mask]
        lp_clean = last_prices[valid_mask]
        # Hit rate počítáme stejně jako v diagnostice pro konzistenci
        hit_pred_std = hit if count >= 2 else np.nan # Pokud count=1, hit je nan
        hold_up = (td_clean > 0).astype(int)
        hit_hold_std = np.mean(hold_up)
        mae_p_std = np.mean(np.abs(td_clean - pd_clean))
        valid_ret_mask_std = (np.abs(lp_clean) > 1e-9)
        if valid_ret_mask_std.sum() > 0:
            lp_valid = lp_clean[valid_ret_mask_std]
            pred_d_valid = pd_clean[valid_ret_mask_std]
            true_d_valid = td_clean[valid_ret_mask_std]
            pred_ret = np.divide(pred_d_valid, lp_valid, out=np.zeros_like(pred_d_valid), where=lp_valid!=0)
            true_ret = np.divide(true_d_valid, lp_valid, out=np.zeros_like(true_d_valid), where=lp_valid!=0)
            mae_r_std = np.mean(np.abs(true_ret - pred_ret))

    return EvalResult(hit_pred_std, hit_hold_std, corr_delta, mae_p_std, mae_r_std, count,
                      corr_price, anti, zeros, mean_prod)

def is_csv_symbol(s):
    return isinstance(s, str) and s.strip().lower().endswith('.csv')


# -----------------------------
# Theta Báze z GPT skriptu (stejná)
# -----------------------------
def generate_theta_basis(time_points, q):
    """Generuje Jacobiho theta bázi θ1..θ4."""
    n_funcs = [1, 2, 3, 4]
    N = len(time_points)
    basis_matrix = np.zeros((N, len(n_funcs)), dtype=float)
    q_mpf = mp.mpf(q)
    for i, t in enumerate(time_points):
         t_mpf = mp.mpf(t)
         for j, n in enumerate(n_funcs):
             try:
                 val = jtheta(n, t_mpf, q_mpf)
                 if mp.isnan(val) or mp.isinf(val) or abs(val) > 1e10: basis_matrix[i, j] = np.nan
                 else: basis_matrix[i, j] = float(val)
             except Exception: basis_matrix[i, j] = np.nan
    return basis_matrix

# -----------------------------
# Jádro evaluace (GPT Logika + Ridge + Predikce DELTY)
# -----------------------------
def evaluate_symbol_csv(path: str,
                        window: int,
                        horizon: int,
                        q_param: float,
                        ema_alpha: float,
                        lam: float,
                        # Nevyužité parametry
                        interval: str, **kwargs # Zachytí zbytek parametrů
                        ):

    df_in = _read_price_csv(path)
    times = df_in["time"].to_numpy()
    closes = df_in["close"].to_numpy().astype(float)
    n_total = len(closes)

    # --- ZMĚNA: Příprava cílové veličiny Y = DELTA ---
    y_target = np.full(n_total, np.nan)
    if n_total > horizon:
        # y_target[t] = closes[t+horizon] - closes[t]
        y_target[:-horizon] = closes[horizon:] - closes[:-horizon]
    # --- KONEC ZMĚNY ---

    # 1. Výpočet celé theta báze B_all (θ1..θ4)
    time_scale_factor = 100.0
    t_idx_scaled = np.arange(n_total, dtype=float) / time_scale_factor
    print(f"Generating theta basis with q={q_param} for {n_total} points...")
    B_all = generate_theta_basis(t_idx_scaled, q_param)
    M = B_all.shape[1]
    if np.isnan(B_all).any():
         print(f"Warning: NaN values found in basis matrix for {path}. Replacing with zeros.")
         B_all = np.nan_to_num(B_all)

    rows = []
    preds_delta, trues_delta, lasts_price = [], [], []

    # 2. Walk-forward predikce (s Ridge na Deltu)
    start_idx = window
    if start_idx >= n_total - horizon:
         needed = window + horizon
         raise RuntimeError(f"Nedostatek dat ({n_total}) pro window={window} a horizon={horizon}.")

    print(f"Starting Walk-Forward Ridge (Delta Target) from index t0={start_idx} up to {n_total - horizon - 1}")

    for t0 in range(start_idx, n_total - horizon):
        # --- KAUZÁLNĚ ČISTÁ LOGIKA ---
        # Fit koeficientů 'beta' pomocí dat do času t0 včetně
        hi_fit = t0 + 1
        lo_fit = max(0, hi_fit - window)
        current_window_size = hi_fit - lo_fit

        min_samples_needed = max(8, M + 1)
        if current_window_size < min_samples_needed: continue

        # Příprava dat pro fit
        Phi_fit = B_all[lo_fit:hi_fit, :]
        # --- ZMĚNA: Cíl je y_target (delta) ---
        y_fit = y_target[lo_fit:hi_fit]
        # --- KONEC ZMĚNY ---

        valid_fit_mask = ~np.isnan(y_fit) & ~np.isnan(Phi_fit).any(axis=1)
        Phi_fit_clean = Phi_fit[valid_fit_mask, :]
        y_fit_clean = y_fit[valid_fit_mask]
        n_clean_fit = len(y_fit_clean)

        if n_clean_fit < min_samples_needed: continue

        # --- APLIKACE RIDGE ---
        if ema_alpha > 1e-9:
            # Časové váhy počítáme jen pro validní body
            time_indices_in_window = np.arange(current_window_size)
            valid_time_indices = time_indices_in_window[valid_fit_mask] # Indexy validních bodů v původním okně
            # Váhy počítáme relativně k t0 (hi_fit - 1)
            time_dist = valid_time_indices - (current_window_size - 1) # Vzdálenost od konce okna
            time_weights_raw = np.exp(ema_alpha * time_dist) # Váha klesá do minulosti
            time_weights = np.maximum(time_weights_raw / (time_weights_raw.sum() + 1e-12), 1e-12)
            time_weights_sqrt = np.sqrt(time_weights)
            Phi_fit_weighted = Phi_fit_clean * time_weights_sqrt[:, None]
            y_fit_weighted = y_fit_clean * time_weights_sqrt
        else:
            Phi_fit_weighted = Phi_fit_clean
            y_fit_weighted = y_fit_clean

        beta = np.zeros(M)
        if M > 0:
            if Phi_fit_weighted.shape[0] >= M:
                 beta = ridge(Phi_fit_weighted, y_fit_weighted, lam)
            else: continue
        # --- KONEC RIDGE ---

        # --- Predikce (použijeme bázi z času t0) ---
        # Na rozdíl od předchozí verze, nyní predikujeme deltu,
        # takže použijeme vstupní bázi z ČASU t0 (b_now).
        b_now = B_all[t0, :]
        if np.isnan(b_now).any(): continue

        # Predikce budoucí DELTY pomocí naučeného 'beta'
        # Předpokládáme, že beta platí pro neváženou bázi b_now
        pred_y_delta = 0.0
        if beta.shape[0] == M:
             try:
                  pred_y_delta = float( b_now @ beta )
             except ValueError: continue
        else: continue

        # --- Výpočet skutečné delty ---
        last_price = closes[t0]
        future_price = closes[t0 + horizon]
        if np.isnan(last_price) or np.isnan(future_price): continue
        true_delta = future_price - last_price

        # Použijeme pred_y_delta přímo jako predikovanou deltu
        pred_delta = pred_y_delta

        if np.isnan(pred_delta) or not np.isfinite(pred_delta): continue

        # Uložení výsledků
        pred_dir = int(np.sign(pred_delta))
        true_dir = int(np.sign(true_delta))
        correct_pred_val = 1 if pred_dir != 0 and pred_dir == true_dir else 0

        rows.append({
            "time": str(times[t0]), "entry_idx": int(t0), "compare_idx": int(t0 + horizon),
            "last_price": float(last_price),
            "pred_price": float(last_price + pred_delta), # Odhad budoucí ceny
            "future_price": float(future_price),
            "pred_delta": float(pred_delta),
            "true_delta": float(true_delta),
            "pred_dir": pred_dir, "true_dir": true_dir, "correct_pred": correct_pred_val,
        })
        preds_delta.append(pred_delta)
        trues_delta.append(true_delta)
        lasts_price.append(last_price)

    # Konec smyčky

    if not rows:
        print(f"Warning: No valid prediction rows produced for {path}.")
        return pd.DataFrame([]), {}

    df_eval = pd.DataFrame(rows)

    # --- Ukládání a metriky ---
    base_key = _safe_stem_key(path)
    out_eval_csv = f"eval_h_{base_key}_gptRidgeDelta.csv" # Odlišíme název
    Path(out_eval_csv).parent.mkdir(parents=True, exist_ok=True)
    wanted_cols = [
        "time","entry_idx","compare_idx", "last_price","pred_price","future_price",
        "pred_delta","true_delta","pred_dir","true_dir","correct_pred"
    ]
    existing_eval_cols = [c for c in wanted_cols if c in df_eval.columns]
    df_eval[existing_eval_cols].to_csv(out_eval_csv, index=False, float_format='%.8f')
    print(f"\nUloženo CSV: {out_eval_csv}")

    # Metriky (s diagnostikou)
    res = metrics_with_diag(np.array(lasts_price, dtype=float),
                            np.array(preds_delta, dtype=float),
                            np.array(trues_delta, dtype=float))

    # summary JSON
    summary = {
        "symbol": path, "interval": interval, "window": window, "horizon": horizon,
        "q_param": q_param, "lambda": lam, "ema_alpha": ema_alpha,
        "target_type": "delta", # Explicitně uvedeme cíl
        "hit_rate_pred": getattr(res, 'hit_rate_pred', float('nan')),
        "hit_rate_hold": getattr(res, 'hit_rate_hold', float('nan')),
        "corr_pred_true": getattr(res, 'corr_pred_true', float('nan')),
        "mae_delta": getattr(res, 'mae_delta', float('nan')),
        "mae_return": getattr(res, 'mae_return', float('nan')),
        "count": int(getattr(res, 'count', 0)),
        "corr_price": getattr(res, 'corr_price', float('nan')),
        "anti_hit_rate": getattr(res, 'anti_hit_rate', float('nan')),
        "zero_rate": getattr(res, 'zero_rate', float('nan')),
        "mean_delta_prod": getattr(res, 'mean_delta_prod', float('nan')),
    }
    out_json = f"sum_h_{base_key}_gptRidgeDelta.json" # Odlišíme název
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    try:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.integer)): return int(obj)
                elif isinstance(obj, (np.float_, np.floating)): return float(obj) if np.isfinite(obj) else None
                elif isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        with open(out_json, "w") as f:
            summary_serializable = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in summary.items()}
            json.dump(summary_serializable, f, indent=2, cls=NpEncoder)
        print(f"Uloženo summary: {out_json}")
    except Exception as e:
        print(f"Error saving JSON summary for {path}: {e}")

    # log výstup
    print("\n--- HSTRATEGY vs HOLD (GPT Logic + Ridge on Delta) ---")
    def fmt(val): return f"{val:9.6f}" if val is not None and np.isfinite(val) else "   nan   "
    print(f"hit_rate_pred:  {fmt(summary['hit_rate_pred'])}")
    print(f"hit_rate_hold:  {fmt(summary['hit_rate_hold'])}")
    print(f"corr_pred_true (Δ): {fmt(summary['corr_pred_true'])}")
    print(f"mae_delta:      {fmt(summary['mae_delta'])}")
    print(f"mae_return:     {fmt(summary['mae_return'])}")
    print(f"count:          {summary['count']}\n")
    print(f"Diag corr_price: {fmt(summary['corr_price'])}")
    print(f"Diag anti_hit:   {fmt(summary['anti_hit_rate'])}")
    print(f"Diag zero_rate:  {fmt(summary['zero_rate'])}")

    return df_eval, summary


def run_batch(symbol_paths: str, **eval_kwargs):
    """Spustí evaluaci pro všechny symboly."""
    syms = [s.strip() for s in symbol_paths.split(",") if s.strip()]
    rows = []
    global ARGS
    for sym in syms:
        print(f"\n=== Running {sym} ===\n")
        try:
            df_eval, summary = evaluate_symbol_csv(sym, **eval_kwargs)
            if summary: rows.append(summary)
        except RuntimeError as re: print(f"RUNTIME ERROR processing {sym}: {re}"); rows.append({"symbol": sym, "error": str(re), **eval_kwargs})
        except ValueError as ve: print(f"VALUE ERROR processing {sym}: {ve}"); rows.append({"symbol": sym, "error": str(ve), **eval_kwargs})
        except Exception as e: print(f"UNEXPECTED ERROR processing {sym}: {e}"); rows.append({"symbol": sym, "error": f"Unexpected: {e}", **eval_kwargs})
    return pd.DataFrame(rows)

# -----------------------------
# CLI (Upraveno pro GPT + Ridge + Delta)
# -----------------------------
def parse_args():
    """Parsování argumentů."""
    p = argparse.ArgumentParser(description="GPT Theta Logic with Ridge Regression on Delta.")
    p.add_argument("--symbols", required=True, help="CSV paths, comma-separated")
    p.add_argument("--csv-time-col", default='time')
    p.add_argument("--csv-close-col", default='close')
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--q", dest="q_param", type=float, default=0.5)
    p.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    p.add_argument("--ema-alpha", type=float, default=0.0)
    p.add_argument("--out", required=True, help="Summary CSV output path.")
    p.add_argument("--interval", default="1h")
    p.add_argument("--phase", default="theta_gpt_ridge_delta") # Označení metody
    # Dummy args
    p.add_argument("--baseP", type=float, default=0)
    p.add_argument("--sigma", type=float, default=0)
    p.add_argument("--N-even", type=int, default=0)
    p.add_argument("--N-odd", type=int, default=0)
    # p.add_argument("--target-type", default="delta") # Cíl je fixní
    p.add_argument("--pred-ensemble", default="N/A")
    p.add_argument("--max-by", default="N/A")
    return p.parse_args()

def main():
    """Hlavní funkce."""
    global ARGS
    ARGS = parse_args()

    eval_kwargs = {
        "interval": ARGS.interval, "window": ARGS.window, "horizon": ARGS.horizon,
        "q_param": ARGS.q_param, "ema_alpha": ARGS.ema_alpha, "lam": ARGS.lam,
        # Dummy
        "baseP": 0, "sigma": 0, "N-even": 0, "N-odd": 0, "target_type": "delta",
        "pred_ensemble": "N/A", "max_by": "N/A",
    }

    out_path = Path(ARGS.out)
    out_dir = out_path.parent
    try: out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e: print(f"Warning: Error creating output directory '{out_dir}': {e}")

    df_summary = run_batch(ARGS.symbols, **eval_kwargs)

    if not df_summary.empty:
        success_mask = ~df_summary.apply(lambda row: 'error' in row and pd.notna(row['error']), axis=1)
        success_rows = df_summary[success_mask].copy()

        if not success_rows.empty:
            cols_out = [
                  "symbol", "window", "horizon", "q_param", "lambda", "ema_alpha", "target_type",
                  "hit_rate_pred", "hit_rate_hold", "delta_hit",
                  "corr_pred_true", "mae_delta", "mae_return", "count",
                  "corr_price", "anti_hit_rate", "zero_rate", "mean_delta_prod"
            ]
            if 'lam' in success_rows.columns:
                 success_rows = success_rows.rename(columns={"lam": "lambda"})
            if 'hit_rate_pred' in success_rows.columns and 'hit_rate_hold' in success_rows.columns:
                 success_rows.loc[:, 'delta_hit'] = success_rows['hit_rate_pred'] - success_rows['hit_rate_hold']
            else:
                 success_rows.loc[:, 'delta_hit'] = np.nan

            existing_cols = [c for c in cols_out if c in success_rows.columns]
            out_df = success_rows[existing_cols]

            try:
                out_df.to_csv(out_path, index=False, float_format='%.6f')
                print(f"\nUloženo (pouze úspěšné běhy): {out_path}")
                print(out_df.to_string(index=False, float_format='%.6f', na_rep='NaN'))
            except Exception as e: print(f"Error saving summary CSV to '{out_path}': {e}")

        error_rows = df_summary[~success_mask]
        if not error_rows.empty:
            print("\nErrors occurred during processing:")
            print(error_rows[['symbol', 'error']].to_string(index=False))
    else:
        print("[warn] No records produced.")

if __name__ == "__main__":
    main()
