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
# Helpery (kombinace z obou světů)
# -----------------------------

def _read_price_csv(path: str) -> pd.DataFrame:
    """Načte CSV a najde sloupce 'time' a 'close'."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at '{path}'")
        raise
    except Exception as e:
        print(f"ERROR: Failed to read CSV '{path}': {e}")
        raise

    tcol = None
    time_candidates = ["time", "timestamp", "date", "datetime", "Time", "Timestamp", "open_time"]
    for c in time_candidates:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        print(f"Warning: Time column not found in '{path}'. Using row index.")
        df["time"] = np.arange(len(df))
        tcol = "time"

    ccol = None
    close_candidates = ["close", "Close", "price", "Price", "close_price"]
    open_candidates = ["open", "Open", "open_price"]
    for c in close_candidates:
        if c in df.columns:
            ccol = c
            break
    if ccol is None:
        for c_open in open_candidates:
             if c_open in df.columns:
                  print(f"Warning: 'close' column not found in '{path}'. Using '{c_open}' as fallback.")
                  ccol = c_open
                  break
        if ccol is None:
             raise ValueError(f"'{path}': Nenašel jsem sloupec s cenou ('close' ani 'open'). Columns: {list(df.columns)}")


    out = df[[tcol, ccol]].copy()
    out.columns = ["time", "close"]

    try:
        time_parsed = pd.to_datetime(out['time'], utc=True, errors='coerce')
        if time_parsed.isnull().any() and pd.api.types.is_numeric_dtype(df[tcol]):
             # print(f"Info: Parsing time column in '{path}' as milliseconds since epoch.")
             time_parsed = pd.to_datetime(df[tcol], unit='ms', utc=True, errors='coerce')
        if time_parsed.isnull().any():
             print(f"Warning: Time parsing failed for some rows in '{path}'. Falling back to row index for consistency.")
             out['time'] = pd.to_datetime(np.arange(len(out)), unit='s', utc=True)
        else:
             out['time'] = time_parsed
    except Exception as e:
         print(f"Warning: Could not parse time column reliably in '{path}': {e}. Falling back to row index.")
         out['time'] = pd.to_datetime(np.arange(len(out)), unit='s', utc=True)

    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    out = out.dropna(subset=['time', 'close']).sort_values('time').reset_index(drop=True)
    out = out.drop_duplicates(subset=['time'], keep='first').reset_index(drop=True)

    if len(out) == 0:
        raise ValueError(f"No valid numeric 'close' data found in '{path}' after cleaning and parsing.")
    return out

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Bezpečný výpočet korelace."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mask = ~np.isnan(a) & ~np.isnan(b)
    a_clean, b_clean = a[mask], b[mask]
    n_clean = len(a_clean)
    if n_clean < 2: return float("nan")
    var_a = np.var(a_clean)
    var_b = np.var(b_clean)
    if var_a < 1e-12 or var_b < 1e-12: return 0.0
    try:
        corr_matrix = np.corrcoef(a_clean, b_clean)
        if isinstance(corr_matrix, np.ndarray) and corr_matrix.shape == (2, 2):
            return float(np.clip(corr_matrix[0, 1], -1.0, 1.0))
        else: return float("nan")
    except Exception: return float("nan")

def _safe_stem_key(path: str) -> str:
    """Vytvoří bezpečný klíč ze jména souboru."""
    stem = Path(path).stem
    safe_key = "".join(c if c.isalnum() else '_' for c in stem).upper()
    max_len = 50
    return safe_key[:max_len] + '_TRUNC' if len(safe_key) > max_len else safe_key

@dataclass
class EvalResult:
    """Struktura pro ukládání výsledků metrik."""
    hit_rate_pred: float
    hit_rate_hold: float
    corr_pred_true: float # Korelace DELTA
    mae_delta: float      # MAE DELTA
    mae_return: float
    count: int
    # Diagnostika
    corr_price: float = float('nan')
    anti_hit_rate: float = float('nan')
    zero_rate: float = float('nan')
    mean_delta_prod: float = float('nan')


def metrics_with_diag(last_prices, pred_delta, true_delta):
    """Vypočítá metriky výkonu VČETNĚ diagnostiky."""
    pred_delta = np.asarray(pred_delta).ravel()
    true_delta = np.asarray(true_delta).ravel()
    last_prices = np.asarray(last_prices).ravel()

    # Maska validních hodnot pro delty a ceny
    valid_mask = ~np.isnan(pred_delta) & ~np.isnan(true_delta) & ~np.isnan(last_prices)
    count = valid_mask.sum()

    if count < 2: # Potřebujeme alespoň 2 body pro korelaci
        print("\n--- DEBUG DIAGNOSTICS ---")
        print("[DBG] Not enough valid data points for diagnostics.")
        print("-------------------------\n")
        return EvalResult(float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), count)

    # Čistá data
    pd_clean = pred_delta[valid_mask]
    td_clean = true_delta[valid_mask]
    lp_clean = last_prices[valid_mask]

    # Dopočítáme ceny pro diagnostiku
    pp_clean = lp_clean + pd_clean
    fp_clean = lp_clean + td_clean

    # --- Diagnostika ---
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
    # --- Konec Diagnostiky ---

    # Standardní metriky (počítané znovu pro jistotu)
    mask_nonzero_true_std = (sign_true != 0)
    hit_pred_std = np.mean(correct) if total_valid_compare > 0 else np.nan # Hit rate ze všech validních
    # Nebo jen z non-zero true? Použijeme stejnou definici jako v diag bloku
    # hit_pred_std = hit

    hold_up = (td_clean > 0).astype(int)
    hit_hold_std = np.mean(hold_up) if len(hold_up) > 0 else np.nan

    mae_p_std = np.mean(np.abs(td_clean - pd_clean))

    valid_ret_mask_std = (np.abs(lp_clean) > 1e-9)
    mae_r_std = float('nan')
    if valid_ret_mask_std.sum() > 0:
        lp_valid = lp_clean[valid_ret_mask_std]
        pred_d_valid = pd_clean[valid_ret_mask_std]
        true_d_valid = td_clean[valid_ret_mask_std]
        pred_ret = np.divide(pred_d_valid, lp_valid, out=np.zeros_like(pred_d_valid), where=lp_valid!=0)
        true_ret = np.divide(true_d_valid, lp_valid, out=np.zeros_like(true_d_valid), where=lp_valid!=0)
        mae_r_std = np.mean(np.abs(true_ret - pred_ret))

    return EvalResult(hit, hit_hold_std, corr_delta, mae_p_std, mae_r_std, count,
                      corr_price, anti, zeros, mean_prod)

def is_csv_symbol(s):
    return isinstance(s, str) and s.strip().lower().endswith('.csv')


# -----------------------------
# Theta Báze z GPT skriptu
# -----------------------------
def generate_theta_basis(time_points, q):
    """
    Vygeneruje hodnoty Jacobiho theta funkcí θ1 až θ4 pro dané časové body.
    Návrat: numpy pole tvaru (len(time_points), 4).
    """
    n_funcs = [1, 2, 3, 4]
    N = len(time_points)
    basis_matrix = np.zeros((N, len(n_funcs)), dtype=float)
    q_mpf = mp.mpf(q) # Převedeme q na mpf jednou
    # Vektorizovaný výpočet času pro mpmath (může být pomalé)
    # Zkusíme iterativně
    for i, t in enumerate(time_points):
         t_mpf = mp.mpf(t) # Převedeme čas na mpf
         for j, n in enumerate(n_funcs):
             try:
                 # Použijeme z=t, nome=q
                 val = jtheta(n, t_mpf, q_mpf)
                 # Kontrola na extrémní hodnoty nebo NaN z mpmath
                 if mp.isnan(val) or mp.isinf(val) or abs(val) > 1e10:
                      basis_matrix[i, j] = np.nan # Nahradíme problematické hodnoty
                 else:
                      basis_matrix[i, j] = float(val)
             except Exception as e:
                 print(f"Warning: mpmath.jtheta failed at t={t}, n={n} with q={q}: {e}")
                 basis_matrix[i, j] = np.nan # Označíme jako neplatné
    return basis_matrix

# Gram-Schmidt zůstává stejný - není potřeba pro Variant A
# def gram_schmidt_orthonormal(vecs): ...

# -----------------------------
# Jádro evaluace (GPT Varianta A - Direct Extrapolation)
# -----------------------------

def evaluate_symbol_csv(path: str,
                        window: int,
                        horizon: int,
                        q_param: float,
                        # Nevyužité parametry (pro kompatibilitu s run_batch)
                        interval: str, baseP: float, sigma: float, N_even: int, N_odd: int,
                        target_type: str, ema_alpha: float, lam: float,
                        pred_ensemble: str, max_by: str
                        ):

    df_in = _read_price_csv(path)
    times = df_in["time"].to_numpy() # Pro ukládání časů
    closes = df_in["close"].to_numpy().astype(float)
    n_total = len(closes)

    # 1. Výpočet celé theta báze B_all (θ1..θ4)
    # Použijeme jednoduchý index jako časovou osu pro jtheta
    # Můžeme škálovat, např. vydělit nějakou konstantou, aby argumenty nebyly příliš velké
    time_scale_factor = 100.0 # Experimentální faktor
    t_idx_scaled = np.arange(n_total, dtype=float) / time_scale_factor
    print(f"Generating theta basis with q={q_param} for {n_total} points...")
    B_all = generate_theta_basis(t_idx_scaled, q_param)
    M = B_all.shape[1] # Počet bázových funkcí (mělo by být 4)

    # Kontrola NaN v bázi
    if np.isnan(B_all).any():
         print(f"Warning: NaN values found in generated basis matrix for {path}. Replacing with zeros.")
         B_all = np.nan_to_num(B_all) # Nahradí NaN nulami

    rows = []
    preds_delta, trues_delta, lasts_price = [], [], []

    # 2. Walk-forward predikce (Varianta A)
    # Potřebujeme 'window' historie pro fit koeficientů 'c'
    # Predikujeme pro čas t0+horizon pomocí 'c' naučeného do času t0
    start_idx = window # První index t0, kde máme dost historie pro fit
    if start_idx >= n_total - horizon:
         needed = window + horizon
         raise RuntimeError(f"Nedostatek dat ({n_total}) pro window={window} a horizon={horizon}. Potřeba alespoň {needed}.")

    print(f"Starting Direct Extrapolation from index t0={start_idx} up to {n_total - horizon - 1}")

    for t0 in range(start_idx, n_total - horizon):
        # --- KAUZÁLNĚ ČISTÁ LOGIKA ---
        # Fit koeficientů 'c' pomocí dat do času t0 včetně
        hi_fit = t0 + 1 # Horní mez pro slicing (exkluzivní)
        lo_fit = max(0, hi_fit - window) # Zajistíme, že nejdeme pod 0
        current_window_size = hi_fit - lo_fit

        # Potřebujeme alespoň M vzorků pro lstsq (počet bázových funkcí)
        if current_window_size < M:
            # print(f"Skipping t0={t0}: Not enough samples ({current_window_size}) < {M} features.")
            continue

        # Příprava dat pro fit
        Phi_fit = B_all[lo_fit:hi_fit, :]
        y_fit = closes[lo_fit:hi_fit]

        # Kontrola validity dat pro fit
        valid_fit_mask = ~np.isnan(y_fit) & ~np.isnan(Phi_fit).any(axis=1)
        Phi_fit_clean = Phi_fit[valid_fit_mask, :]
        y_fit_clean = y_fit[valid_fit_mask]
        n_clean_fit = len(y_fit_clean)

        if n_clean_fit < M: # Musíme mít alespoň tolik bodů, kolik je sloupců báze
            # print(f"Skipping t0={t0}: Not enough clean samples ({n_clean_fit}) < {M} features for lstsq.")
            continue

        # Fit pomocí least squares (lstsq)
        try:
            # rcond=None použije default stroje, můžeme nastavit malou hodnotu pro stabilitu
            c, residuals, rank, s = np.linalg.lstsq(Phi_fit_clean, y_fit_clean, rcond=-1) # rcond=-1 pro potlačení varování
            c = c.reshape(-1) # Zajistíme 1D tvar
            # Kontrola ranku pro stabilitu (volitelné)
            # if rank < M: print(f"Warning: Rank deficient fit at t0={t0}. Rank={rank}<{M}")
        except np.linalg.LinAlgError:
            # print(f"Skipping t0={t0}: lstsq solver failed.")
            continue
        except ValueError as ve:
             print(f"Skipping t0={t0}: ValueError in lstsq setup ({ve}). Shapes: Phi={Phi_fit_clean.shape}, y={y_fit_clean.shape}")
             continue


        # --- Predikce (použijeme bázi z budoucího času t0 + horizon) ---
        idx_future = t0 + horizon
        b_future = B_all[idx_future, :]

        if np.isnan(b_future).any():
             # print(f"Skipping t0={t0}: NaN in future basis vector at t0+h={idx_future}.")
             continue

        # Predikce budoucí ÚROVNĚ ceny
        pred_y_future_price = 0.0
        if c.shape[0] == M: # Kontrola tvaru koeficientů
             try:
                  pred_y_future_price = float( b_future @ c )
             except ValueError:
                  # print(f"Skipping t0={t0}: ValueError during prediction dot product.")
                  continue
        else: # Neočekávaný tvar 'c'
             # print(f"Skipping t0={t0}: Coefficient vector 'c' shape mismatch ({c.shape[0]} vs {M}).")
             continue


        # --- Výpočet delty ---
        last_price = closes[t0] # Cena v čase t0, ke které se vztahuje delta
        if np.isnan(last_price):
             # print(f"Skipping t0={t0}: NaN in last_price at t0={t0}.")
             continue

        pred_delta = pred_y_future_price - last_price

        # Skutečná budoucí cena a delta
        future_price = closes[t0 + horizon]
        if np.isnan(future_price):
             # print(f"Skipping t0={t0}: NaN in future_price at t0+h={t0+horizon}.")
             continue
        true_delta = future_price - last_price

        # Finální kontrola platnosti
        if np.isnan(pred_delta) or np.isnan(true_delta) or not np.isfinite(pred_delta) or not np.isfinite(true_delta):
            # print(f"Skipping t0={t0}: Invalid final pred_delta ({pred_delta}) or true_delta ({true_delta}).")
            continue

        # Uložení výsledků
        pred_dir = int(np.sign(pred_delta))
        true_dir = int(np.sign(true_delta))
        correct_pred_val = 1 if pred_dir != 0 and pred_dir == true_dir else 0

        rows.append({
            "time": str(times[t0]),
            "entry_idx": int(t0),
            "compare_idx": int(t0 + horizon),
            "last_price": float(last_price),
            "pred_price": float(pred_y_future_price), # Ukládáme predikovanou úroveň
            "future_price": float(future_price),
            "pred_delta": float(pred_delta),
            "true_delta": float(true_delta),
            "pred_dir": pred_dir,
            "true_dir": true_dir,
            "correct_pred": correct_pred_val,
        })
        preds_delta.append(pred_delta)
        trues_delta.append(true_delta)
        lasts_price.append(last_price)

    # Konec smyčky walk-forward

    if not rows:
        print(f"Warning: No valid prediction rows produced for {path}.")
        # ... (zbytek chybové hlášky) ...
        return pd.DataFrame([]), {}


    df_eval = pd.DataFrame(rows)

    # --- Ukládání a metriky ---
    base_key = _safe_stem_key(path)
    # Odlišíme název souboru
    out_eval_csv = f"eval_h_{base_key}_gptA.csv"
    Path(out_eval_csv).parent.mkdir(parents=True, exist_ok=True)
    wanted_cols = [
        "time","entry_idx","compare_idx",
        "last_price","pred_price","future_price",
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
        "q_param": q_param, # Přidán q parametr
        # Odebrány parametry z Jacobiho fixed leak
        "hit_rate_pred": getattr(res, 'hit_rate_pred', float('nan')),
        "hit_rate_hold": getattr(res, 'hit_rate_hold', float('nan')),
        "corr_pred_true": getattr(res, 'corr_pred_true', float('nan')), # Korelace DELTA
        "mae_delta": getattr(res, 'mae_price', float('nan')), # Přejmenováno pro jasnost
        "mae_return": getattr(res, 'mae_return', float('nan')),
        "count": int(getattr(res, 'count', 0)),
        # Diagnostika
        "corr_price": getattr(res, 'corr_price', float('nan')),
        "anti_hit_rate": getattr(res, 'anti_hit_rate', float('nan')),
        "zero_rate": getattr(res, 'zero_rate', float('nan')),
        "mean_delta_prod": getattr(res, 'mean_delta_prod', float('nan')),
    }
    out_json = f"sum_h_{base_key}_gptA.json" # Odlišíme název
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    try:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj) if np.isfinite(obj) else None
                elif isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        with open(out_json, "w") as f:
            summary_serializable = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in summary.items()}
            json.dump(summary_serializable, f, indent=2, cls=NpEncoder)
        print(f"Uloženo summary: {out_json}")
    except Exception as e:
        print(f"Error saving JSON summary for {path}: {e}")

    # log výstup do konzole
    print("\n--- HSTRATEGY vs HOLD (GPT Logic - Variant A) ---")
    def fmt(val): return f"{val:9.6f}" if val is not None and np.isfinite(val) else "   nan   "
    print(f"hit_rate_pred:  {fmt(summary['hit_rate_pred'])}")
    print(f"hit_rate_hold:  {fmt(summary['hit_rate_hold'])}")
    print(f"corr_pred_true (Δ): {fmt(summary['corr_pred_true'])}")
    print(f"mae_delta:      {fmt(summary['mae_delta'])}")
    print(f"mae_return:     {fmt(summary['mae_return'])}")
    print(f"count:          {summary['count']}\n")
    # Výpis diagnostiky
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
            # Předáme ARGS pro přístup k csv_time_col, csv_close_col
            df_eval, summary = evaluate_symbol_csv(sym, **eval_kwargs)
            if summary: rows.append(summary)
        except RuntimeError as re: print(f"RUNTIME ERROR processing {sym}: {re}"); rows.append({"symbol": sym, "error": str(re), **eval_kwargs})
        except ValueError as ve: print(f"VALUE ERROR processing {sym}: {ve}"); rows.append({"symbol": sym, "error": str(ve), **eval_kwargs})
        except Exception as e: print(f"UNEXPECTED ERROR processing {sym}: {e}"); rows.append({"symbol": sym, "error": f"Unexpected: {e}", **eval_kwargs})

    return pd.DataFrame(rows)

# -----------------------------
# CLI (Upraveno pro GPT logiku)
# -----------------------------

def parse_args():
    """Parsování argumentů příkazové řádky."""
    p = argparse.ArgumentParser(description="GPT Theta Logic Evaluator (Direct Extrapolation).")
    p.add_argument("--symbols", required=True, help="CSV paths, comma-separated")
    p.add_argument("--csv-time-col", default='time', help="Time column name (default: time)")
    p.add_argument("--csv-close-col", default='close', help="Close column name (default: close)")
    # Model Params
    p.add_argument("--window", type=int, default=256, help="Rolling window size for lstsq fit.")
    p.add_argument("--horizon", type=int, default=4, help="Prediction horizon.")
    p.add_argument("--q", type=float, default=0.5, help="q parameter for Jacobi theta functions (0 < q < 1).")
    # Odstraněny parametry z předchozích verzí
    # p.add_argument("--baseP", type=float, default=36.0)
    # p.add_argument("--sigma", type=float, default=0.8)
    # p.add_argument("--N-even", type=int, default=6)
    # p.add_argument("--N-odd", type=int, default=6)
    # p.add_argument("--target-type", choices=['logprice', 'logret', 'delta'], default='delta')
    # p.add_argument("--ema-alpha", type=float, default=0.0)
    # p.add_argument("--lambda", dest="lam", type=float, default=1e-3)

    p.add_argument("--out", required=True, help="Summary CSV output path.")
    # Ponechány pro info/kompatibilitu
    p.add_argument("--interval", default="1h")
    p.add_argument("--phase", default="theta_gpt_A") # Označení metody
    return p.parse_args()


def main():
    """Hlavní funkce skriptu."""
    global ARGS
    ARGS = parse_args()

    # Parametry specifické pro tuto metodu
    eval_kwargs = {
        "interval": ARGS.interval,
        "window": ARGS.window,
        "horizon": ARGS.horizon,
        "q_param": ARGS.q,
        # Tyto parametry zde nejsou potřeba, ale předáme je, aby evaluate_symbol_csv mělo všechny argumenty
        "baseP": 0, "sigma": 0, "N_even": 0, "N_odd": 0,
        "target_type": "N/A", "ema_alpha": 0, "lam": 0,
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
            # Sloupce pro výstupní CSV (upravené pro GPT logiku)
            cols_out = [
                  "symbol", "window", "horizon", "q_param",
                  "hit_rate_pred", "hit_rate_hold", "delta_hit",
                  "corr_pred_true", "mae_delta", "mae_return", "count",
                  # Diagnostika
                  "corr_price", "anti_hit_rate", "zero_rate", "mean_delta_prod"
            ]
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
            except Exception as e:
                print(f"Error saving summary CSV to '{out_path}': {e}")

        error_rows = df_summary[~success_mask]
        if not error_rows.empty:
            print("\nErrors occurred during processing:")
            print(error_rows[['symbol', 'error']].to_string(index=False))

    else:
        print("[warn] No records produced (did you pass valid CSV paths or did all runs fail?)")

if __name__ == "__main__":
    main()
