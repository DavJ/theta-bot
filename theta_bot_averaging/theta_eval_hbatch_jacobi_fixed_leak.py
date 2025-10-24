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
# Původní Helpery (z jacobi.py)
# -----------------------------

def _read_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # zkus různé názvy času
    tcol = None
    for c in ["time", "timestamp", "date", "datetime", "Time", "Timestamp"]:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        # fallback: index na pořadí
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

    # Robustnější parsování času z fetch_csv
    try:
        if pd.api.types.is_numeric_dtype(out['time']):
             out['time'] = pd.to_datetime(out['time'], unit='ms', utc=True, errors='coerce')
        else:
             out['time'] = pd.to_datetime(out['time'], utc=True, errors='coerce')
    except Exception as e:
         print(f"Warning: Could not parse time column reliably: {e}. Falling back to row index.")
         out['time'] = pd.to_datetime(np.arange(len(out)), unit='s', utc=True) # Fallback čas

    out = out.dropna(subset=['time', 'close']).sort_values('time').reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(f"No valid data found in '{path}' after cleaning.")
    return out

def _make_period_grid(minP: int, maxP: int, nP: int) -> np.ndarray:
    # log-space mřížka period (z původního jacobi.py)
    if nP <= 0: return np.array([])
    if nP == 1: return np.array([minP]) if minP==maxP else np.array([(minP+maxP)/2])
    return np.unique(np.round(np.geomspace(minP, maxP, nP)).astype(int))

def ridge(X, y, lam): # Zkopírováno z původního kódu
    XT = X.T
    A = XT @ X
    n = A.shape[0]
    # Přidání regularizace na diagonálu
    if n > 0:
        A.flat[::n+1] += lam
    b = XT @ y
    try:
        # Preferujeme solve pro stabilitu
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback na lstsq, pokud je matice singulární
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return beta

def _corr(a: np.ndarray, b: np.ndarray) -> float: # Zkopírováno z původního kódu
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    # Pokud je některý vektor konstantní, korelace je 0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    # Ošetření NaN hodnot
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return float("nan")
    a_clean, b_clean = a[mask], b[mask]
    if np.allclose(a_clean, a_clean[0]) or np.allclose(b_clean, b_clean[0]):
        return 0.0
    return float(np.corrcoef(a_clean, b_clean)[0, 1])

def _safe_stem_key(path: str) -> str: # Zkopírováno z původního kódu
    base = Path(path).name
    stem = Path(path).stem
    return stem.replace(".", "").upper() # Použijeme stem pro konzistenci s původním kódem

@dataclass
class EvalResult: # Definovano v opravené verzi
    hit_rate_pred: float
    hit_rate_hold: float
    corr_pred_true: float
    mae_price: float
    mae_return: float
    count: int

def metrics(last_prices, pred_delta, true_delta): # Definovano v opravené verzi
    if len(true_delta) == 0:
        return EvalResult(float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0)

    pred_dir = np.sign(pred_delta)
    true_dir = np.sign(true_delta)
    mask_nonzero_true = (true_dir != 0) & ~np.isnan(pred_dir) & ~np.isnan(true_dir)
    if mask_nonzero_true.sum() > 0:
         hit_pred = (pred_dir[mask_nonzero_true] == true_dir[mask_nonzero_true]).mean()
    else:
         hit_pred = float('nan')

    hold_up = (true_delta > 0).astype(int)
    mask_valid_hold = ~np.isnan(hold_up)
    hit_hold = hold_up[mask_valid_hold].mean() if mask_valid_hold.sum() > 0 else float('nan')

    c = _corr(pred_delta, true_delta) # Použijeme opravenou _corr

    valid_mae = ~np.isnan(true_delta) & ~np.isnan(pred_delta)
    mae_p = np.mean(np.abs(true_delta[valid_mae] - pred_delta[valid_mae])) if valid_mae.sum() > 0 else float('nan')

    valid_ret = valid_mae & (np.abs(last_prices) > 1e-9)
    pred_ret = np.full_like(pred_delta, np.nan)
    true_ret = np.full_like(true_delta, np.nan)
    pred_ret[valid_ret] = pred_delta[valid_ret] / last_prices[valid_ret]
    true_ret[valid_ret] = true_delta[valid_ret] / last_prices[valid_ret]
    mae_r = np.mean(np.abs(true_ret[valid_ret] - pred_ret[valid_ret])) if valid_ret.sum() > 0 else float('nan')

    return EvalResult(hit_pred, hit_hold, c, mae_p, mae_r, valid_mae.sum())

def is_csv_symbol(s): # Pro jistotu
    return s.lower().endswith('.csv')

# -----------------------------
# Nová Theta Báze (z opravené verze)
# -----------------------------

def build_theta_q_basis(t_idx: np.ndarray, baseP: float, sigma: float, N_even: int, N_odd: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_idx, dtype=float)
    N = len(t)
    # Zajištění, že sigma není nula, aby se zabránilo dělení nulou nebo log(0)
    sigma = max(sigma, 1e-9)
    q = np.exp(-np.pi * sigma)
    # Zajištění, že baseP není nula
    baseP = max(baseP, 1e-9)
    omega = 2.0 * np.pi / baseP
    z = omega * t

    cols = []
    q_weights = []

    for k in range(1, N_even + 1):
        q_pow_k2 = np.abs(q)**(k**2)
        cols.append(np.cos(2 * k * z))
        q_weights.append(2 * q_pow_k2)
        cols.append((-1)**k * np.cos(2 * k * z))
        q_weights.append(2 * q_pow_k2)

    for m in range(N_odd):
        idx_float = m + 0.5
        q_pow_m2 = np.abs(q)**(idx_float**2)
        angle = (2 * m + 1) * z
        cols.append(np.cos(angle))
        q_weights.append(2 * q_pow_m2)
        cols.append((-1)**m * np.sin(angle))
        q_weights.append(2 * q_pow_m2)

    if not cols:
        return np.zeros((N, 0)), np.zeros(0)

    B_raw = np.stack(cols, axis=1)
    q_weights = np.array(q_weights, dtype=float)
    q_weights = np.maximum(q_weights, 1e-12) # Ošetření velmi malých vah
    return B_raw, q_weights

def weighted_qr_simple(B: np.ndarray, q_weights: np.ndarray, time_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    W, D = B.shape
    if D == 0: return np.zeros((W, 0)), np.zeros((0, 0))

    # Kontrola vah
    q_weights = np.maximum(q_weights, 1e-12)
    time_weights = np.maximum(time_weights, 1e-12)

    B_weighted_cols = B * np.sqrt(q_weights)[None, :]
    B_weighted_rows_cols = B_weighted_cols * np.sqrt(time_weights)[:, None]

    try:
         Q, R = np.linalg.qr(B_weighted_rows_cols)
    except np.linalg.LinAlgError:
         print("Warning: QR decomposition failed, returning zeros.")
         Q = np.zeros_like(B)
         R = np.zeros((D,D))
    return Q, R

# -----------------------------
# Jádro evaluace (OPRAVENÉ - BEZ LEAKU)
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
                        max_by: str): # Nyní nevyužito

    df_in = _read_price_csv(path) # Použijeme _read_price_csv
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
    elif target_type == 'delta':
         if len(closes) > horizon:
             y_target[:-horizon] = closes[horizon:] - closes[:-horizon]
    else:
        raise ValueError(f"Neznámý target_type: {target_type}")

    # 2. Výpočet celé theta báze
    t_idx = np.arange(len(closes), dtype=float)
    B_all_raw, q_weights = build_theta_q_basis(t_idx, baseP, sigma, N_even, N_odd)
    D = B_all_raw.shape[1]

    rows = []
    preds_delta, trues_delta, lasts_price = [], [], []

    # 3. Walk-forward evaluace
    start_idx = window + 1 # Index prvního bodu, pro který můžeme trénovat
    if start_idx >= len(closes) - horizon:
         raise RuntimeError(f"Nedostatek dat ({len(closes)}) pro window={window} a horizon={horizon}.")

    for t0 in range(start_idx, len(closes) - horizon):
        # --- OPRAVENÁ LOGIKA: TRÉNINK KONČÍ PŘED BODEM PREDIKCE ---
        hi_train = t0
        lo_train = hi_train - window
        if lo_train < 0: lo_train = 0
        current_window_size = hi_train - lo_train

        if current_window_size < 8: continue

        # --- Příprava trénovacích dat ---
        B_win_raw = B_all_raw[lo_train:hi_train, :]
        y_win = y_target[lo_train:hi_train]

        # Kontroly validity dat
        if np.isnan(y_win).any() or np.isnan(B_win_raw).any():
             continue
        if B_win_raw.shape[0] != current_window_size or y_win.shape[0] != current_window_size:
             continue

        # --- Vážená regrese ---
        time_weights = np.exp(ema_alpha * (np.arange(current_window_size) - (current_window_size - 1)))
        time_weights = np.maximum(time_weights / time_weights.sum(), 1e-12) # Normalizace a ošetření

        # Přeskočíme, pokud váhy nejsou platné
        if np.isnan(time_weights).any() or time_weights.sum() < 1e-9:
            continue

        B_win_raw_weighted_rows = B_win_raw * np.sqrt(time_weights)[:, None]
        y_win_weighted_rows = y_win * np.sqrt(time_weights)

        # Ošetření q_weights - měly by mít dimenzi D
        if q_weights.shape[0] != D:
             print(f"Warning: q_weights shape mismatch ({q_weights.shape[0]} vs {D}) at t0={t0}. Skipping.")
             continue
        q_weights_sqrt = np.sqrt(q_weights)
        # Použijeme broadcasting pro aplikaci vah na sloupce
        if D > 0:
             B_win_fully_weighted = B_win_raw_weighted_rows * q_weights_sqrt[None, :]
        else:
             B_win_fully_weighted = B_win_raw_weighted_rows # Pokud je D=0

        # Ridge
        if B_win_fully_weighted.shape[1] == 0: # Pokud nejsou žádné features
             beta_raw_weighted = np.array([])
             pred_y = 0.0 # Nebo jiná default hodnota
        elif B_win_fully_weighted.shape[0] < B_win_fully_weighted.shape[1]: # Více features než bodů - nelze řešit
             print(f"Warning: Underdetermined system (features > samples) at t0={t0}. Skipping.")
             continue
        else:
            try:
                 beta_raw_weighted = ridge(B_win_fully_weighted, y_win_weighted_rows, lam)
            except np.linalg.LinAlgError:
                 print(f"Warning: Ridge solver failed at t0={t0}. Skipping.")
                 continue

        # --- Predikce ---
        b_raw_now = B_all_raw[t0, :]
        if D > 0:
             pred_y = float( (b_raw_now * q_weights_sqrt) @ beta_raw_weighted )
        else:
             pred_y = 0.0

        # --- Převod predikce y na predikci delty ---
        last_price = closes[t0]
        if np.isnan(last_price): continue # Přeskočíme, pokud aktuální cena chybí

        pred_delta = np.nan # Default na NaN
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

        # Skutečná budoucí cena a delta
        future_price = closes[t0 + horizon]
        if np.isnan(future_price): continue # Přeskočíme, pokud budoucí cena chybí
        true_delta = future_price - last_price

        # Přeskočíme, pokud predikce selhala
        if np.isnan(pred_delta): continue

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

    if not rows:
        raise RuntimeError(f"No rows produced for {path}. Check data length vs. window/horizon.")

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
    df_eval[wanted_cols].to_csv(out_eval_csv, index=False, float_format='%.8f')
    print(f"\nUloženo CSV: {out_eval_csv}")

    # Metriky
    res = metrics(np.array(lasts_price, dtype=float), np.array(preds_delta, dtype=float), np.array(trues_delta, dtype=float))

    # summary JSON
    summary = {
        "symbol": path, "interval": interval, "window": window, "horizon": horizon,
        "baseP": baseP, "sigma": sigma, "N_even": N_even, "N_odd": N_odd,
        "target_type": target_type, "ema_alpha": ema_alpha, "lambda": lam,
        "hit_rate_pred": res.hit_rate_pred, "hit_rate_hold": res.hit_rate_hold,
        "corr_pred_true": res.corr_pred_true, "mae_price": res.mae_price,
        "mae_return": res.mae_return, "count": res.count,
    }
    out_json = f"sum_h_{base_key}.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Uloženo summary: {out_json}")
    except Exception as e:
        print(f"Error saving JSON summary: {e}")

    # log výstup do konzole
    print("\n--- HSTRATEGY vs HOLD (Theta Q Basis - Corrected) ---")
    print(f"hit_rate_pred:  {res.hit_rate_pred:9.6f}")
    print(f"hit_rate_hold:  {res.hit_rate_hold:9.6f}")
    print(f"corr_pred_true: {res.corr_pred_true:9.6f}")
    print(f"mae_price (delta): {res.mae_price:9.6f}")
    print(f"mae_return:     {res.mae_return:9.6f}")
    print(f"count:          {res.count}\n")

    return df_eval, summary


def run_batch(symbol_paths: str, **eval_kwargs):
    syms = [s.strip() for s in symbol_paths.split(",") if s.strip()]
    rows = []
    # Předáme ARGS globálně pro fetch_csv (není ideální, ale rychlé řešení)
    global ARGS
    for sym in syms:
        print(f"\n=== Running {sym} ===\n")
        try:
            # Předáme ARGS pro přístup k csv_time_col, csv_close_col
            df_eval, summary = evaluate_symbol_csv(sym, **eval_kwargs)
            rows.append(summary)
        except Exception as e:
            print(f"ERROR processing {sym}: {e}")
            rows.append({"symbol": sym, "error": str(e)}) # Záznam chyby

    df = pd.DataFrame(rows)
    return df

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Theta Q-Basis Evaluator - Leak Corrected.")
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
    p.add_argument("--ema-alpha", type=float, default=0.1, help="Decay factor for time weights (0=uniform).")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-3, help="Ridge regularization strength.")
    # Legacy/Unused (pro konzistenci s voláním)
    p.add_argument("--pred-ensemble", default="avg")
    p.add_argument("--max-by", default="transform")
    p.add_argument("--interval", default="1h")
    p.add_argument("--phase", default="theta_q")

    p.add_argument("--out", required=True, help="Summary CSV output path.")
    return p.parse_args()


def main():
    global ARGS # Abychom měli přístup v evaluate_symbol_csv a run_batch
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

    # výstupní tabulka
    if not df_summary.empty and 'error' not in df_summary.columns[0].lower(): # Kontrola prvního sloupce na 'error'
         cols_out = [
              "symbol", "target_type", "window", "horizon", "baseP", "sigma", "N_even", "N_odd", "lambda",
              "hit_rate_pred", "hit_rate_hold", "corr_pred_true", "mae_price", "mae_return", "count"
         ]
         if 'hit_rate_pred' in df_summary.columns and 'hit_rate_hold' in df_summary.columns:
              df_summary['delta_hit'] = df_summary['hit_rate_pred'] - df_summary['hit_rate_hold']
              cols_out.insert(11, 'delta_hit')

         # Přejmenování lambda pro výstupní CSV (pokud je potřeba)
         df_summary = df_summary.rename(columns={"lambda": "ridge_lambda"})
         if "ridge_lambda" in df_summary.columns and "lambda" not in cols_out:
              cols_out[cols_out.index("N_odd")+1] = "ridge_lambda" # Nahradíme 'lambda' v seznamu

         # Zajistíme, že všechny sloupce existují, než je vybereme
         existing_cols = [c for c in cols_out if c in df_summary.columns]
         out_df = df_summary[existing_cols] # Použijeme jen existující sloupce

         out_path = ARGS.out
         out_dir = Path(out_path).parent
         out_dir.mkdir(parents=True, exist_ok=True)
         out_df.to_csv(out_path, index=False, float_format='%.6f')
         print(f"\nUloženo: {out_path}")
         # Výpis s omezením na existující sloupce
         print(out_df.to_string(index=False, float_format='%.6f'))

    elif not df_summary.empty:
         print("\nErrors occurred during processing:")
         # Vytiskneme jen řádky s chybami
         error_rows = df_summary[df_summary.apply(lambda row: 'error' in row and pd.notna(row['error']), axis=1)]
         if not error_rows.empty:
              print(error_rows.to_string(index=False))
         else:
              print("No specific error messages recorded, but summary might be incomplete.")
    else:
        print("[warn] No records written to summary (did you pass valid CSV paths?)")


if __name__ == "__main__":
    main()
