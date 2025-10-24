#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

# -----------------------------
# Původní Helpery (z jacobi.py) + Nové + Opravené
# -----------------------------

def _read_price_csv(path: str) -> pd.DataFrame:
    """Načte CSV a najde sloupce 'time' a 'close'."""
    df = pd.read_csv(path)
    # zkus různé názvy času
    tcol = None
    for c in ["time", "timestamp", "date", "datetime", "Time", "Timestamp"]:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        # fallback: index na pořadí
        print(f"Warning: Time column not found in '{path}'. Using row index.")
        df["time"] = np.arange(len(df))
        tcol = "time"
    # zkus různé názvy closu
    ccol = None
    for c in ["close", "Close", "price", "Price", "close_price"]:
        if c in df.columns:
            ccol = c
            break
    if ccol is None:
        raise ValueError(f"'{path}': nenašel jsem sloupec s close cenou.")

    out = df[[tcol, ccol]].copy()
    out.columns = ["time", "close"]

    # Robustnější parsování času
    try:
        if pd.api.types.is_numeric_dtype(out['time']):
             # Předpokládáme milisekundy, pokud je číslo
             out['time'] = pd.to_datetime(out['time'], unit='ms', utc=True, errors='coerce')
        else:
             out['time'] = pd.to_datetime(out['time'], utc=True, errors='coerce')
    except Exception as e:
         print(f"Warning: Could not parse time column reliably in '{path}': {e}. Falling back to row index.")
         out['time'] = pd.to_datetime(np.arange(len(out)), unit='s', utc=True) # Fallback čas jako sekundy

    out = out.dropna(subset=['time', 'close']).sort_values('time').reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(f"No valid data found in '{path}' after cleaning.")
    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    out = out.dropna(subset=['close'])
    if len(out) == 0:
        raise ValueError(f"No valid numeric 'close' data found in '{path}'.")
    return out

def _make_period_grid(minP: int, maxP: int, nP: int) -> np.ndarray:
    """Vytvoří log-space mřížku period (zde se nepoužívá, ale pro úplnost)."""
    if nP <= 0: return np.array([])
    if nP == 1: return np.array([minP]) if minP==maxP else np.array([(minP+maxP)/2.0])
    # Zajistíme, aby minP a maxP byly kladné pro logaritmus
    minP_safe = max(minP, 1e-6)
    maxP_safe = max(maxP, 1e-6)
    if minP_safe >= maxP_safe: return np.array([minP_safe])
    return np.unique(np.round(np.geomspace(minP_safe, maxP_safe, nP)).astype(int))

def ridge(X, y, lam):
    """Řeší Ridge regresi (X'X + lambda*I)beta = X'y."""
    if X.shape[0] < X.shape[1]:
        print(f"Warning: Underdetermined system ({X.shape[0]} samples, {X.shape[1]} features). Ridge might be unstable.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        print("Warning: Empty matrix X passed to ridge.")
        return np.zeros(X.shape[1]) # Vrací nulový vektor správné dimenze
    XT = X.T
    A = XT @ X
    n = A.shape[0]
    # Přidání regularizace na diagonálu
    if n > 0:
        A.flat[::n+1] += lam
    b = XT @ y
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Warning: np.linalg.solve failed in ridge. Falling back to lstsq.")
        try:
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            print("ERROR: lstsq also failed in ridge. Returning zeros.")
            beta = np.zeros(n)
    return beta

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Bezpečný výpočet korelace."""
    a = np.asarray(a)
    b = np.asarray(b)
    mask = ~np.isnan(a) & ~np.isnan(b)
    a_clean, b_clean = a[mask], b[mask]
    if len(a_clean) < 2:
        return float("nan")
    # Pokud je některý vektor konstantní po vyčištění NaN
    if np.allclose(a_clean, a_clean[0]) or np.allclose(b_clean, b_clean[0]):
        return 0.0
    try:
        return float(np.corrcoef(a_clean, b_clean)[0, 1])
    except Exception:
        return float("nan") # Fallback pro neočekávané chyby

def _safe_stem_key(path: str) -> str:
    """Vytvoří bezpečný klíč ze jména souboru pro výstupní soubory."""
    stem = Path(path).stem
    # Odstraníme problematické znaky a převedeme na velká písmena
    safe_key = "".join(c if c.isalnum() else '_' for c in stem).upper()
    return safe_key

@dataclass
class EvalResult:
    """Struktura pro ukládání výsledků metrik."""
    hit_rate_pred: float
    hit_rate_hold: float
    corr_pred_true: float
    mae_price: float
    mae_return: float
    count: int
    corr_shuffle: float = float('nan') # Přidáno pro sanity check
    corr_lag1: float = float('nan')    # Přidáno pro sanity check

def metrics(last_prices, pred_delta, true_delta):
    """Vypočítá metriky výkonu a sanity checky."""
    if len(true_delta) == 0:
        return EvalResult(float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0)

    pred_delta = np.asarray(pred_delta)
    true_delta = np.asarray(true_delta)
    last_prices = np.asarray(last_prices)

    # Základní metriky
    pred_dir = np.sign(pred_delta)
    true_dir = np.sign(true_delta)
    mask_nonzero_true = (true_dir != 0) & ~np.isnan(pred_dir) & ~np.isnan(true_dir)
    count = mask_nonzero_true.sum()
    if count > 0:
         hit_pred = (pred_dir[mask_nonzero_true] == true_dir[mask_nonzero_true]).mean()
    else:
         hit_pred = float('nan')

    hold_up = (true_delta > 0).astype(int)
    mask_valid_hold = ~np.isnan(hold_up)
    hit_hold = hold_up[mask_valid_hold].mean() if mask_valid_hold.sum() > 0 else float('nan')

    c = _corr(pred_delta, true_delta)

    # MAE (pouze na validních párech)
    valid_mae = mask_nonzero_true # Použijeme stejnou masku pro konzistenci
    mae_p = np.mean(np.abs(true_delta[valid_mae] - pred_delta[valid_mae])) if count > 0 else float('nan')

    # MAE Návratnosti
    valid_ret = valid_mae & (np.abs(last_prices[mask_nonzero_true]) > 1e-9)
    mae_r = float('nan')
    if valid_ret.sum() > 0:
        # Indexujeme původní pole pomocí masky, abychom dostali správné last_prices
        lp_valid = last_prices[mask_nonzero_true][valid_ret[mask_nonzero_true]]
        pred_d_valid = pred_delta[mask_nonzero_true][valid_ret[mask_nonzero_true]]
        true_d_valid = true_delta[mask_nonzero_true][valid_ret[mask_nonzero_true]]
        pred_ret = pred_d_valid / lp_valid
        true_ret = true_d_valid / lp_valid
        mae_r = np.mean(np.abs(true_ret - pred_ret))


    # --- SANITY TESTS ---
    corr_shuffle = float('nan')
    corr_lag1 = float('nan')

    if count > 1:
        pred_clean = pred_delta[mask_nonzero_true]
        true_clean = true_delta[mask_nonzero_true]

        # Shuffle Test
        rng = np.random.default_rng(12345)
        shuffled_true = true_clean.copy()
        rng.shuffle(shuffled_true)
        corr_shuffle = _corr(pred_clean, shuffled_true)

        # Lag Test
        if count > 2:
            pred_t = pred_clean[:-1]
            true_t_plus_1 = true_clean[1:]
            corr_lag1 = _corr(pred_t, true_t_plus_1)

    print(f"Sanity Check - Shuffle Corr: {corr_shuffle:.4f}")
    print(f"Sanity Check - Lag-1 Corr:   {corr_lag1:.4f}")
    # --- KONEC SANITY TESTS ---

    return EvalResult(hit_pred, hit_hold, c, mae_p, mae_r, count, corr_shuffle, corr_lag1)

def is_csv_symbol(s):
    """Kontroluje, zda string končí na .csv (case-insensitive)."""
    return isinstance(s, str) and s.strip().lower().endswith('.csv')

# -----------------------------
# Theta Báze z Jacobiho funkcí
# -----------------------------

def build_theta_q_basis(t_idx: np.ndarray, baseP: float, sigma: float, N_even: int, N_odd: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vytvoří základní (neortogonalizovanou) bázi z q-řad Jacobiho thet.
    Používá z(t) = omega * t.
    Vrací: Matice báze B_raw [N, D], Vektor q-vah [D]
    """
    t = np.asarray(t_idx, dtype=float)
    N = len(t)
    sigma = max(sigma, 1e-9) # Ošetření sigma=0
    q = np.exp(-np.pi * sigma)
    baseP = max(baseP, 1e-9) # Ošetření baseP=0
    omega = 2.0 * np.pi / baseP
    z = omega * t

    cols = []
    q_weights = []

    # Theta 3 & Theta 0 (sudé cos)
    for k in range(1, N_even + 1):
        q_pow_k2 = np.abs(q)**(k**2)
        if q_pow_k2 < 1e-12: break # Optimalizace - zanedbatelné váhy
        term_cos = np.cos(2 * k * z)
        # Theta 3
        cols.append(term_cos)
        q_weights.append(2 * q_pow_k2)
        # Theta 0
        cols.append((-1)**k * term_cos)
        q_weights.append(2 * q_pow_k2)

    # Theta 2 (cos) & Theta 1 (sin) (liché)
    for m in range(N_odd): # m jde od 0 do N_odd-1
        idx_float = m + 0.5
        q_pow_m2 = np.abs(q)**(idx_float**2)
        if q_pow_m2 < 1e-12: break # Optimalizace
        angle = (2 * m + 1) * z
        # Theta 2
        cols.append(np.cos(angle))
        q_weights.append(2 * q_pow_m2)
        # Theta 1
        cols.append((-1)**m * np.sin(angle))
        q_weights.append(2 * q_pow_m2)

    if not cols:
        print("Warning: No basis columns generated. Check N_even, N_odd, sigma.")
        return np.zeros((N, 0)), np.zeros(0)

    B_raw = np.stack(cols, axis=1) # Shape (N, D)
    q_weights = np.array(q_weights, dtype=float)
    q_weights = np.maximum(q_weights, 1e-12) # Ošetření velmi malých vah

    return B_raw, q_weights

# Funkce weighted_qr_simple není přímo potřeba, pokud děláme váženou regresi

# -----------------------------
# Jádro evaluace (OPRAVENÉ - KAUZÁLNĚ ČISTÉ)
# -----------------------------

def evaluate_symbol_csv(path: str,
                        interval: str,
                        window: int,
                        horizon: int,
                        baseP: float,
                        sigma: float,
                        N_even: int,
                        N_odd: int,
                        target_type: str,
                        ema_alpha: float,
                        lam: float,
                        pred_ensemble: str,
                        max_by: str): # pred_ensemble, max_by nyní méně relevantní

    df_in = _read_price_csv(path)
    times = df_in["time"].to_numpy()
    closes = df_in["close"].to_numpy().astype(float)

    # 1. Příprava cílové veličiny (y_target)
    y_target = np.full_like(closes, np.nan) # Inicializace NaN
    if target_type == 'logprice':
        y_target = np.log(np.maximum(closes, 1e-9))
    elif target_type == 'logret':
        if len(closes) > 1:
            log_closes = np.log(np.maximum(closes, 1e-9))
            y_target[1:] = np.diff(log_closes)
            y_target[0] = np.nan # První hodnota je neplatná
    elif target_type == 'delta':
         if len(closes) > horizon:
             y_target[:-horizon] = closes[horizon:] - closes[:-horizon]
    else:
        raise ValueError(f"Neznámý target_type: {target_type}")

    # 2. Výpočet celé theta báze
    t_idx = np.arange(len(closes), dtype=float)
    B_all_raw, q_weights = build_theta_q_basis(t_idx, baseP, sigma, N_even, N_odd)
    D = B_all_raw.shape[1]
    q_weights_sqrt = np.sqrt(q_weights) if D > 0 else np.array([])

    rows = []
    preds_delta, trues_delta, lasts_price = [], [], []

    # 3. Walk-forward evaluace
    # start_idx: první index 't0', pro který můžeme udělat predikci
    # Potřebujeme 'window' historie končící v 't0-1' A cíl pro 't0+horizon'.
    start_idx = window # Nejčasnější možný konec trénovacího okna je 'window-1'
    if start_idx >= len(closes) - horizon:
         needed = window + horizon
         raise RuntimeError(f"Nedostatek dat ({len(closes)}) pro window={window} a horizon={horizon}. Potřeba alespoň {needed}.")

    print(f"Starting walk-forward from index {start_idx} to {len(closes) - horizon -1}")

    for t0 in range(start_idx, len(closes) - horizon):
        # --- KAUZÁLNĚ ČISTÁ LOGIKA ---
        # Predikce se dělá pro čas t0 (použije vstup B_all_raw[t0,:])
        # Trénink musí použít data pouze do času t0-1.
        hi_train = t0  # Horní index pro slicing (exkluzivní)
        lo_train = hi_train - window
        if lo_train < 0: lo_train = 0
        current_window_size = hi_train - lo_train

        # Musíme mít dost bodů pro trénink
        if current_window_size < max(8, D): # Min 8 bodů a alespoň tolik bodů kolik features
            # print(f"Skipping t0={t0}: Not enough samples ({current_window_size}) for {D} features.")
            continue

        # --- Příprava trénovacích dat ---
        B_win_raw = B_all_raw[lo_train:hi_train, :]
        y_win = y_target[lo_train:hi_train]

        # Kontroly validity dat
        valid_mask_train = ~np.isnan(y_win) & ~np.isnan(B_win_raw).any(axis=1)
        B_win_raw_clean = B_win_raw[valid_mask_train, :]
        y_win_clean = y_win[valid_mask_train]
        n_clean = len(y_win_clean)

        if n_clean < max(8, D): # Kontrola po odstranění NaN
            # print(f"Skipping t0={t0}: Not enough clean samples ({n_clean}) for {D} features.")
            continue

        # --- Vážená regrese ---
        # Použijeme časové váhy jen pro validní body
        time_weights = np.exp(ema_alpha * (np.arange(n_clean) - (n_clean - 1)))
        time_weights = np.maximum(time_weights / time_weights.sum(), 1e-12)

        if np.isnan(time_weights).any() or time_weights.sum() < 1e-9:
            # print(f"Skipping t0={t0}: Invalid time weights.")
            continue

        B_win_clean_weighted_rows = B_win_raw_clean * np.sqrt(time_weights)[:, None]
        y_win_clean_weighted_rows = y_win_clean * np.sqrt(time_weights)

        if D > 0:
             if q_weights_sqrt.shape[0] != D:
                 print(f"ERROR: q_weights shape mismatch ({q_weights_sqrt.shape[0]} vs {D}) at t0={t0}.")
                 continue
             B_win_fully_weighted = B_win_clean_weighted_rows * q_weights_sqrt[None, :]
        else:
             B_win_fully_weighted = B_win_clean_weighted_rows # Prázdná matice

        # Ridge
        beta_raw_weighted = np.zeros(D) # Default
        if D > 0:
            try:
                 # Přidána kontrola rozměrů pro ridge
                 if B_win_fully_weighted.shape[0] >= D:
                      beta_raw_weighted = ridge(B_win_fully_weighted, y_win_clean_weighted_rows, lam)
                 else:
                      # print(f"Skipping t0={t0}: Underdetermined system after cleaning ({B_win_fully_weighted.shape[0]}<{D}).")
                      continue
            except np.linalg.LinAlgError:
                 # print(f"Skipping t0={t0}: Ridge solver failed.")
                 continue
            except ValueError as ve: # Zachytí chyby tvaru matic
                 print(f"Skipping t0={t0}: ValueError in ridge setup ({ve}). Shapes: X={B_win_fully_weighted.shape}, y={y_win_clean_weighted_rows.shape}")
                 continue


        # --- Predikce (použijeme data z času t0) ---
        b_raw_now = B_all_raw[t0, :]
        if np.isnan(b_raw_now).any(): # Kontrola vstupu pro predikci
             # print(f"Skipping t0={t0}: NaN in features at prediction time.")
             continue

        pred_y = 0.0
        if D > 0:
             try:
                  pred_y = float( (b_raw_now * q_weights_sqrt) @ beta_raw_weighted )
             except ValueError as ve: # Chyba při násobení matic
                  print(f"Skipping t0={t0}: ValueError during prediction ({ve}). Shapes: b_now={b_raw_now.shape}, q_sqrt={q_weights_sqrt.shape}, beta={beta_raw_weighted.shape}")
                  continue

        # --- Převod predikce y na predikci delty ---
        last_price = closes[t0]
        if np.isnan(last_price):
             # print(f"Skipping t0={t0}: NaN in last_price.")
             continue

        pred_delta = np.nan
        try:
             if target_type == 'logprice':
                 pred_future_logprice = pred_y
                 pred_future_price = np.exp(pred_future_logprice)
                 pred_delta = pred_future_price - last_price
             elif target_type == 'logret':
                 pred_logret = pred_y
                 pred_future_price = last_price * np.exp(pred_logret)
                 pred_delta = pred_future_price - last_price
             elif target_type == 'delta':
                  pred_delta = pred_y
        except OverflowError:
             # print(f"Skipping t0={t0}: Overflow during prediction conversion (exp).")
             continue


        # Skutečná budoucí cena a delta
        future_price = closes[t0 + horizon]
        if np.isnan(future_price):
             # print(f"Skipping t0={t0}: NaN in future_price.")
             continue
        true_delta = future_price - last_price

        # Přeskočíme, pokud predikce nebo cíl není platný
        if np.isnan(pred_delta) or np.isnan(true_delta) or np.isinf(pred_delta) or np.isinf(true_delta):
            # print(f"Skipping t0={t0}: Invalid pred_delta ({pred_delta}) or true_delta ({true_delta}).")
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
            "pred_price": float(last_price + pred_delta),
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
        raise RuntimeError(f"No valid rows produced for {path}. Check data quality and parameters.")

    df_eval = pd.DataFrame(rows)

    # --- Ukládání a metriky ---
    base_key = _safe_stem_key(path)
    out_eval_csv = f"eval_h_{base_key}.csv"
    Path(out_eval_csv).parent.mkdir(parents=True, exist_ok=True)
    wanted_cols = [
        "time","entry_idx","compare_idx",
        "last_price","pred_price","future_price",
        "pred_delta","true_delta","pred_dir","true_dir","correct_pred"
    ]
    # Uložíme jen sloupce, které existují, pro případ chyby
    existing_eval_cols = [c for c in wanted_cols if c in df_eval.columns]
    df_eval[existing_eval_cols].to_csv(out_eval_csv, index=False, float_format='%.8f')
    print(f"\nUloženo CSV: {out_eval_csv}")

    # Metriky
    res = metrics(np.array(lasts_price, dtype=float), np.array(preds_delta, dtype=float), np.array(trues_delta, dtype=float))

    # summary JSON
    summary = {
        "symbol": path, "interval": interval, "window": window, "horizon": horizon,
        "baseP": baseP, "sigma": sigma, "N_even": N_even, "N_odd": N_odd,
        "target_type": target_type, "ema_alpha": ema_alpha, "lambda": lam,
        # Použijeme getattr pro bezpečné získání atributů
        "hit_rate_pred": getattr(res, 'hit_rate_pred', float('nan')),
        "hit_rate_hold": getattr(res, 'hit_rate_hold', float('nan')),
        "corr_pred_true": getattr(res, 'corr_pred_true', float('nan')),
        "mae_price": getattr(res, 'mae_price', float('nan')),
        "mae_return": getattr(res, 'mae_return', float('nan')),
        "count": int(getattr(res, 'count', 0)), # Oprava pro JSON serializaci
        "corr_shuffle": getattr(res, 'corr_shuffle', float('nan')),
        "corr_lag1": getattr(res, 'corr_lag1', float('nan')),
    }
    out_json = f"sum_h_{base_key}.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    try:
        # Použijeme vlastní serializátor pro NumPy typy
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2, cls=NpEncoder)
        print(f"Uloženo summary: {out_json}")
    except Exception as e:
        print(f"Error saving JSON summary for {path}: {e}")

    # log výstup do konzole
    print("\n--- HSTRATEGY vs HOLD (Theta Q Basis - Corrected) ---")
    print(f"hit_rate_pred:  {summary['hit_rate_pred']:9.6f}")
    print(f"hit_rate_hold:  {summary['hit_rate_hold']:9.6f}")
    print(f"corr_pred_true: {summary['corr_pred_true']:9.6f}")
    print(f"mae_price (delta): {summary['mae_price']:9.6f}")
    print(f"mae_return:     {summary['mae_return']:9.6f}")
    print(f"count:          {summary['count']}\n")
    print(f"Sanity Shuffle Corr: {summary['corr_shuffle']:.4f}")
    print(f"Sanity Lag-1 Corr:   {summary['corr_lag1']:.4f}")


    return df_eval, summary


def run_batch(symbol_paths: str, **eval_kwargs):
    """Spustí evaluaci pro všechny symboly."""
    syms = [s.strip() for s in symbol_paths.split(",") if s.strip()]
    rows = []
    global ARGS # Nutné pro přístup k ARGS v evaluate_symbol_csv přes _read_price_csv
    for sym in syms:
        print(f"\n=== Running {sym} ===\n")
        try:
            df_eval, summary = evaluate_symbol_csv(sym, **eval_kwargs)
            rows.append(summary)
        except Exception as e:
            print(f"ERROR processing {sym}: {e}")
            # Záznam chyby pro finální report
            rows.append({"symbol": sym,
                         "error": str(e),
                         # Přidáme klíčové parametry pro kontext
                         "window": eval_kwargs.get("window"),
                         "horizon": eval_kwargs.get("horizon"),
                         "baseP": eval_kwargs.get("baseP"),
                         "lambda": eval_kwargs.get("lam")})

    df = pd.DataFrame(rows)
    return df

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    """Parsování argumentů příkazové řádky."""
    p = argparse.ArgumentParser(description="Theta Q-Basis Evaluator - Leak Corrected & Robust.")
    p.add_argument("--symbols", required=True, help="CSV paths, comma-separated")
    p.add_argument("--csv-time-col", default='time', help="Time column name (default: time)")
    p.add_argument("--csv-close-col", default='close', help="Close column name (default: close)")
    # Model Params
    p.add_argument("--window", type=int, default=256, help="Rolling window size.")
    p.add_argument("--horizon", type=int, default=4, help="Prediction horizon.")
    p.add_argument("--baseP", type=float, default=36.0, help="Base period P for z(t)=2pi*t/P.")
    p.add_argument("--sigma", type=float, default=0.8, help="Sigma for q=exp(-pi*sigma).")
    p.add_argument("--N-even", type=int, default=6, help="Truncation N for even theta series.")
    p.add_argument("--N-odd", type=int, default=6, help="Truncation N for odd theta series.")
    p.add_argument("--target-type", choices=['logprice', 'logret', 'delta'], default='delta', help="Target variable for regression.")
    p.add_argument("--ema-alpha", type=float, default=0.0, help="Decay factor for time weights (0=uniform, >0 exp decay).")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-3, help="Ridge regularization strength.")
    # Legacy/Unused - pro konzistenci
    p.add_argument("--pred-ensemble", default="avg", choices=["avg"], help="Ensemble method (only avg supported).")
    p.add_argument("--max-by", default="transform")
    p.add_argument("--interval", default="1h")
    p.add_argument("--phase", default="theta_q")

    p.add_argument("--out", required=True, help="Summary CSV output path.")
    return p.parse_args()


def main():
    """Hlavní funkce skriptu."""
    global ARGS
    ARGS = parse_args()

    eval_kwargs = {
        "interval": ARGS.interval,
        "window": ARGS.window,
        "horizon": ARGS.horizon,
        "baseP": ARGS.baseP,
        "sigma": ARGS.sigma,
        "N_even": ARGS.N_even,
        "N_odd": ARGS.N_odd,
        "target_type": ARGS.target_type,
        "ema_alpha": ARGS.ema_alpha,
        "lam": ARGS.lam,
        "pred_ensemble": ARGS.pred_ensemble,
        "max_by": ARGS.max_by,
    }

    df_summary = run_batch(ARGS.symbols, **eval_kwargs)

    # Příprava finálního výstupního DataFrame
    if not df_summary.empty:
        # Zkontrolujeme, zda první řádek neobsahuje chybu
        if 'error' in df_summary.iloc[0] and pd.notna(df_summary.iloc[0]['error']):
             print("\nErrors occurred during processing:")
             error_rows = df_summary[df_summary.apply(lambda row: 'error' in row and pd.notna(row['error']), axis=1)]
             print(error_rows.to_string(index=False))
             print(f"\nSummary CSV not saved due to errors.")
        else:
            # Sloupce pro výstupní CSV
            cols_out = [
                  "symbol", "target_type", "window", "horizon", "baseP", "sigma",
                  "N_even", "N_odd", "lambda", # Přejmenováno z 'lam'
                  "hit_rate_pred", "hit_rate_hold", "delta_hit", # delta_hit se dopočítá
                  "corr_pred_true", "mae_price", "mae_return", "count",
                  "corr_shuffle", "corr_lag1" # Přidány sanity check metriky
            ]
            # Přejmenování 'lam' na 'lambda' pro výstup
            if 'lam' in df_summary.columns:
                 df_summary = df_summary.rename(columns={"lam": "lambda"})

            # Dopočítání delta_hit
            if 'hit_rate_pred' in df_summary.columns and 'hit_rate_hold' in df_summary.columns:
                 df_summary['delta_hit'] = df_summary['hit_rate_pred'] - df_summary['hit_rate_hold']
            else:
                 df_summary['delta_hit'] = np.nan

            # Zajistíme, že všechny sloupce existují, než je vybereme/seřadíme
            existing_cols = [c for c in cols_out if c in df_summary.columns]
            out_df = df_summary[existing_cols]

            # Uložení a výpis
            out_path = ARGS.out
            out_dir = Path(out_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_path, index=False, float_format='%.6f')
            print(f"\nUloženo: {out_path}")
            print(out_df.to_string(index=False, float_format='%.6f'))

    else:
        print("[warn] No records produced (did you pass valid CSV paths?)")


if __name__ == "__main__":
    main()