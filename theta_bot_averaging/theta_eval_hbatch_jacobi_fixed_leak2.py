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
# Původní Helpery + Nové + Opravené
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

    # zkus různé názvy času
    tcol = None
    # Prioritizujeme 'time', pak 'timestamp', pak další
    time_candidates = ["time", "timestamp", "date", "datetime", "Time", "Timestamp", "open_time"]
    for c in time_candidates:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        print(f"Warning: Time column not found in '{path}' using candidates {time_candidates}. Using row index.")
        df["time"] = np.arange(len(df))
        tcol = "time"

    # zkus různé názvy closu
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

    # Robustnější parsování času
    try:
        # Zkusíme nejprve jako string (ISO formát), errors='coerce' vrátí NaT při chybě
        time_parsed = pd.to_datetime(out['time'], utc=True, errors='coerce')

        # Pokud první pokus selhal (obsahuje NaT) a původní typ byl číslo
        if time_parsed.isnull().any() and pd.api.types.is_numeric_dtype(df[tcol]):
             print(f"Info: Parsing time column in '{path}' as milliseconds since epoch.")
             # Znovu parsujeme původní sloupec jako ms
             time_parsed = pd.to_datetime(df[tcol], unit='ms', utc=True, errors='coerce')

        # Pokud stále selhává, použijeme index
        if time_parsed.isnull().any():
             print(f"Warning: Time parsing failed for some rows in '{path}'. Falling back to row index for failed rows or all if all failed.")
             # Použijeme index jen pro neúspěšné, nebo pro všechny, pokud všechny selhaly? Raději všechny pro konzistenci.
             out['time'] = pd.to_datetime(np.arange(len(out)), unit='s', utc=True)
        else:
             out['time'] = time_parsed

    except Exception as e:
         print(f"Warning: Could not parse time column reliably in '{path}': {e}. Falling back to row index.")
         out['time'] = pd.to_datetime(np.arange(len(out)), unit='s', utc=True)

    # Finální kontrola a čištění
    out['close'] = pd.to_numeric(out['close'], errors='coerce')
    out = out.dropna(subset=['time', 'close']).sort_values('time').reset_index(drop=True)

    # Odstranění duplicitních časů (ponechá první výskyt)
    out = out.drop_duplicates(subset=['time'], keep='first').reset_index(drop=True)

    if len(out) == 0:
        raise ValueError(f"No valid numeric 'close' data found in '{path}' after cleaning and parsing.")
    return out


def _make_period_grid(minP: int, maxP: int, nP: int) -> np.ndarray:
    """Vytvoří log-space mřížku period (zde se nepoužívá, ale pro úplnost)."""
    if nP <= 0: return np.array([])
    if nP == 1: return np.array([minP]) if minP==maxP else np.array([(minP+maxP)/2.0])
    minP_safe = max(minP, 1e-6)
    maxP_safe = max(maxP, 1e-6)
    if minP_safe >= maxP_safe: return np.array([minP_safe])
    # Zajistíme alespoň minimální rozdíl pro geomspace
    if np.isclose(minP_safe, maxP_safe): return np.array([minP_safe])
    try:
        grid = np.geomspace(minP_safe, maxP_safe, nP)
    except ValueError: # Může nastat, pokud jsou vstupy problematické
        grid = np.linspace(minP_safe, maxP_safe, nP) # Fallback na linspace
    return np.unique(np.round(grid).astype(int))

def ridge(X, y, lam):
    """Řeší Ridge regresi (X'X + lambda*I)beta = X'y."""
    if X.ndim != 2 or y.ndim != 1:
         # Pokusíme se převést y na 1D, pokud má tvar (N, 1)
         if y.ndim == 2 and y.shape[1] == 1:
              y = y.ravel()
              if y.ndim != 1:
                   raise ValueError(f"Invalid dimensions for ridge: X={X.shape}, y={y.shape} (cannot flatten y)")
         else:
              raise ValueError(f"Invalid dimensions for ridge: X={X.shape}, y={y.shape}")

    n_samples, n_features = X.shape
    if n_samples < n_features:
        print(f"Warning: Underdetermined system ({n_samples} samples, {n_features} features). Ridge might be unstable.")
    if n_samples == 0 or n_features == 0:
        # print("Warning: Empty matrix X or y passed to ridge.")
        return np.zeros(n_features) # Vrací nulový vektor správné dimenze

    XT = X.T
    A = XT @ X
    # Přidání regularizace na diagonálu
    if n_features > 0:
        A += lam * np.identity(n_features)
    b = XT @ y
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # print("Warning: np.linalg.solve failed in ridge. Falling back to lstsq.")
        try:
            beta = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            # print("ERROR: lstsq also failed in ridge. Returning zeros.")
            beta = np.zeros(n_features)
        except ValueError as ve:
             print(f"ERROR: ValueError in lstsq ({ve}). Shapes: A={A.shape}, b={b.shape}. Returning zeros.")
             beta = np.zeros(n_features)

    # Zajistíme, že beta má vždy správný tvar (1D vektor)
    return beta.reshape(-1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Bezpečný výpočet korelace."""
    a = np.asarray(a).ravel() # Zajistíme 1D tvar
    b = np.asarray(b).ravel()
    mask = ~np.isnan(a) & ~np.isnan(b)
    a_clean, b_clean = a[mask], b[mask]
    n_clean = len(a_clean)
    if n_clean < 2:
        return float("nan")

    # Kontrola konstantních vektorů
    var_a = np.var(a_clean)
    var_b = np.var(b_clean)
    if var_a < 1e-12 or var_b < 1e-12:
        # Pokud jsou oba konstantní a stejné, korelace je 1, jinak 0 nebo NaN? Vraťme 0.
        return 0.0

    try:
        corr_matrix = np.corrcoef(a_clean, b_clean)
        if isinstance(corr_matrix, np.ndarray) and corr_matrix.shape == (2, 2):
            result = corr_matrix[0, 1]
            # Někdy může být výsledek mírně mimo [-1, 1] kvůli numerickým chybám
            return float(np.clip(result, -1.0, 1.0))
        else:
            return float("nan")
    except Exception as e:
        # print(f"Warning: Exception during correlation calculation: {e}")
        return float("nan")


def _safe_stem_key(path: str) -> str:
    """Vytvoří bezpečný klíč ze jména souboru pro výstupní soubory."""
    stem = Path(path).stem
    safe_key = "".join(c if c.isalnum() else '_' for c in stem).upper()
    max_len = 50
    if len(safe_key) > max_len:
         safe_key = safe_key[:max_len] + '_TRUNC'
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
    pred_delta = np.asarray(pred_delta).ravel()
    true_delta = np.asarray(true_delta).ravel()
    last_prices = np.asarray(last_prices).ravel()

    # Maska validních hodnot pro predikce, skutečné hodnoty A last_prices (pro návratnost)
    valid_mask = ~np.isnan(pred_delta) & ~np.isnan(true_delta) & ~np.isnan(last_prices)
    count = valid_mask.sum()

    if count == 0:
        return EvalResult(float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0)

    pred_d_clean = pred_delta[valid_mask]
    true_d_clean = true_delta[valid_mask]
    last_p_clean = last_prices[valid_mask]

    # Základní metriky
    pred_dir = np.sign(pred_d_clean)
    true_dir = np.sign(true_d_clean)
    # Ignorujeme případy, kdy true_dir je 0 pro hit rate predikce
    mask_nonzero_true = (true_dir != 0)
    hit_pred = np.mean(pred_dir[mask_nonzero_true] == true_dir[mask_nonzero_true]) if mask_nonzero_true.sum() > 0 else float('nan')

    # Hit rate hold počítáme na všech validních datech
    hold_up = (true_d_clean > 0).astype(int)
    hit_hold = np.mean(hold_up) if len(hold_up) > 0 else float('nan')

    c = _corr(pred_d_clean, true_d_clean)

    # MAE delty
    mae_p = np.mean(np.abs(true_d_clean - pred_d_clean))

    # MAE Návratnosti
    # Použijeme pouze body, kde last_price není nula
    valid_ret_mask = (np.abs(last_p_clean) > 1e-9)
    mae_r = float('nan')
    if valid_ret_mask.sum() > 0:
        lp_valid = last_p_clean[valid_ret_mask]
        pred_d_valid = pred_d_clean[valid_ret_mask]
        true_d_valid = true_d_clean[valid_ret_mask]
        # Ošetření dělení nulou, i když maska by měla stačit
        pred_ret = np.divide(pred_d_valid, lp_valid, out=np.zeros_like(pred_d_valid), where=lp_valid!=0)
        true_ret = np.divide(true_d_valid, lp_valid, out=np.zeros_like(true_d_valid), where=lp_valid!=0)
        mae_r = np.mean(np.abs(true_ret - pred_ret))


    # --- SANITY TESTS (na vyčištěných datech) ---
    corr_shuffle = float('nan')
    corr_lag1 = float('nan')

    if count > 1:
        # Shuffle Test
        rng = np.random.default_rng(12345)
        shuffled_true = true_d_clean.copy()
        rng.shuffle(shuffled_true)
        corr_shuffle = _corr(pred_d_clean, shuffled_true)

        # Lag Test
        if count > 2:
            pred_t = pred_d_clean[:-1]
            true_t_plus_1 = true_d_clean[1:]
            # Zajistíme stejnou délku pro korelaci
            min_len_lag = min(len(pred_t), len(true_t_plus_1))
            corr_lag1 = _corr(pred_t[:min_len_lag], true_t_plus_1[:min_len_lag])

    # Tyto sanity checky se vypisují v evaluate_symbol_csv
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
    sigma = max(sigma, 1e-9)
    q = np.exp(-np.pi * sigma)
    baseP = max(baseP, 1e-9)
    omega = 2.0 * np.pi / baseP
    z = omega * t

    cols = []
    q_weights = []
    cutoff_thresh = 1e-9 # Zvýšená tolerance pro zanedbání

    # Theta 3 & Theta 0 (sudé cos)
    # k jde od 1 do N_even včetně
    for k in range(1, N_even + 1):
        q_pow_k2 = np.abs(q)**(k**2)
        if q_pow_k2 < cutoff_thresh: break
        term_cos = np.cos(2 * k * z)
        # Theta 3
        cols.append(term_cos)
        q_weights.append(2 * q_pow_k2)
        # Theta 0
        cols.append((-1)**k * term_cos)
        q_weights.append(2 * q_pow_k2)

    # Theta 2 (cos) & Theta 1 (sin) (liché)
    # m jde od 0 do N_odd-1 včetně (celkem N_odd členů)
    for m in range(N_odd):
        idx_float = m + 0.5
        q_pow_m2 = np.abs(q)**(idx_float**2)
        if q_pow_m2 < cutoff_thresh: break
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
    q_weights = np.maximum(q_weights, 1e-12)

    return B_raw, q_weights

# -----------------------------
# Jádro evaluace (FINÁLNĚ OPRAVENÉ - KAUZÁLNĚ ČISTÉ)
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
                        max_by: str):

    df_in = _read_price_csv(path)
    times = df_in["time"].to_numpy()
    closes = df_in["close"].to_numpy().astype(float)
    n_total = len(closes)

    # 1. Příprava cílové veličiny (y_target)
    y_target = np.full(n_total, np.nan)
    if target_type == 'logprice':
        y_target = np.log(np.maximum(closes, 1e-9))
    elif target_type == 'logret':
        if n_total > 1:
            log_closes = np.log(np.maximum(closes, 1e-9))
            y_target = np.concatenate(([np.nan], np.diff(log_closes)))
    elif target_type == 'delta':
         if n_total > horizon:
             # y_target[t] = closes[t+horizon] - closes[t]
             y_target[:-horizon] = closes[horizon:] - closes[:-horizon]
    else:
        raise ValueError(f"Neznámý target_type: {target_type}")

    # 2. Výpočet celé theta báze
    t_idx = np.arange(n_total, dtype=float)
    B_all_raw, q_weights = build_theta_q_basis(t_idx, baseP, sigma, N_even, N_odd)
    D = B_all_raw.shape[1]
    q_weights_sqrt = np.sqrt(q_weights) if D > 0 else np.array([])

    rows = []
    preds_delta, trues_delta, lasts_price = [], [], []

    # 3. Walk-forward evaluace
    # start_idx: první index 't0', pro který můžeme udělat predikci
    # Potřebujeme 'window' historie končící v 't0-1' A cíl pro 't0+horizon'.
    # Nejdřívější t0, kde můžeme trénovat na [0..window-1] je t0=window.
    # Zároveň potřebujeme cíl v t0+horizon, takže smyčka jde do n_total-horizon-1
    start_idx = window
    if start_idx >= n_total - horizon:
         needed = window + horizon + 1 # +1 pro bod predikce
         raise RuntimeError(f"Nedostatek dat ({n_total}) pro window={window} a horizon={horizon}. Potřeba alespoň {needed}.")

    print(f"Starting walk-forward from index t0={start_idx} up to {n_total - horizon - 1}")

    for t0 in range(start_idx, n_total - horizon):
        # --- FINÁLNÍ OPRAVA KAUZALITY ---
        # Predikce se dělá pro čas t0 (použije vstup B_all_raw[t0,:])
        # Trénink musí použít data, kde poslední CÍL byl znám PŘED časem t0.
        # Cíl y_target[t] je znám až v čase t+horizon (nebo později, pokud y_target je logprice/logret).
        # Chceme trénovat na (X[t], y[t]). Poslední cíl y[t_last], který známe v čase t0,
        # je ten, pro který platí t_last + horizon <= t0. Tedy t_last <= t0 - horizon.
        # Horní index pro slicing trénovacích dat (exkluzivní) je tedy t0 - horizon + 1.
        hi_train_corrected = t0 - horizon + 1
        lo_train_corrected = hi_train_corrected - window
        if lo_train_corrected < 0: lo_train_corrected = 0

        current_window_size = hi_train_corrected - lo_train_corrected
        min_samples_needed = max(8, D + 1) # Potřebujeme alespoň D+1 vzorků
        if current_window_size < min_samples_needed:
            # print(f"Skipping t0={t0}: Not enough history ({current_window_size}) < {min_samples_needed}")
            continue
        # --- KONEC FINÁLNÍ OPRAVY ---

        # --- Příprava trénovacích dat (používá opravené indexy) ---
        B_win_raw = B_all_raw[lo_train_corrected:hi_train_corrected, :]
        y_win = y_target[lo_train_corrected:hi_train_corrected]

        # Kontroly validity dat
        valid_mask_train = ~np.isnan(y_win) & ~np.isnan(B_win_raw).any(axis=1)
        B_win_raw_clean = B_win_raw[valid_mask_train, :]
        y_win_clean = y_win[valid_mask_train]
        n_clean = len(y_win_clean)

        if n_clean < min_samples_needed:
            # print(f"Skipping t0={t0}: Not enough clean samples ({n_clean}) < {min_samples_needed}")
            continue

        # --- Vážená regrese ---
        if ema_alpha > 1e-9: # Použijeme threshold místo == 0.0
            # Váhy počítáme relativně ke konci OKNA (tj. k hi_train_corrected)
            time_indices_in_window = np.arange(current_window_size)
            # Nejnovější bod má index current_window_size - 1
            time_weights_raw = np.exp(ema_alpha * (time_indices_in_window - (current_window_size - 1)))
            # Aplikujeme váhy jen na validní body
            time_weights_clean_raw = time_weights_raw[valid_mask_train]
            time_weights = np.maximum(time_weights_clean_raw / (time_weights_clean_raw.sum() + 1e-12), 1e-12)
        else:
            time_weights = np.ones(n_clean) / n_clean # Uniformní váhy

        if np.isnan(time_weights).any() or time_weights.sum() < 1e-9:
            # print(f"Skipping t0={t0}: Invalid time weights.")
            continue
        time_weights_sqrt = np.sqrt(time_weights)

        # Aplikujeme váhy na vyčištěná data
        B_win_clean_weighted_rows = B_win_raw_clean * time_weights_sqrt[:, None]
        y_win_clean_weighted_rows = y_win_clean * time_weights_sqrt

        if D > 0:
             if q_weights_sqrt.shape[0] != D:
                 print(f"ERROR: q_weights shape mismatch ({q_weights_sqrt.shape[0]} vs {D}) at t0={t0}.")
                 continue
             B_win_fully_weighted = B_win_clean_weighted_rows * q_weights_sqrt[None, :]
        else:
             B_win_fully_weighted = B_win_clean_weighted_rows

        # Ridge
        beta_raw_weighted = np.zeros(D)
        if D > 0:
            if B_win_fully_weighted.shape[0] >= D:
                 beta_raw_weighted = ridge(B_win_fully_weighted, y_win_clean_weighted_rows, lam)
            else:
                 # print(f"Skipping t0={t0}: Underdetermined system after cleaning ({B_win_fully_weighted.shape[0]}<{D}).")
                 continue
        # Pokud D=0, beta zůstane [0]

        # --- Predikce (použijeme data z času t0) ---
        b_raw_now = B_all_raw[t0, :]
        if np.isnan(b_raw_now).any():
             # print(f"Skipping t0={t0}: NaN in features at prediction time t0={t0}.")
             continue

        pred_y = 0.0
        if D > 0:
             if beta_raw_weighted.shape[0] == D:
                 try:
                      pred_y = float( (b_raw_now * q_weights_sqrt) @ beta_raw_weighted )
                 except ValueError:
                      # print(f"Skipping t0={t0}: ValueError during prediction dot product.")
                      continue
             else:
                 # print(f"Skipping t0={t0}: beta shape mismatch during prediction.")
                 continue

        # --- Převod predikce y na predikci delty ---
        last_price = closes[t0] # Cena v čase t0, ke kterému se vztahuje predikce
        if np.isnan(last_price) or last_price <= 0:
             # print(f"Skipping t0={t0}: Invalid last_price ({last_price}) at t0={t0}.")
             continue

        pred_delta = np.nan
        try:
             if target_type == 'logprice':
                 # pred_y je odhad log(C[t0+h])
                 pred_future_price = np.exp(pred_y)
                 if not np.isfinite(pred_future_price): raise OverflowError("exp overflow")
                 pred_delta = pred_future_price - last_price # Delta vůči C[t0]
             elif target_type == 'logret':
                 # pred_y je odhad log(C[t0+h]/C[t0+h-1]) - POZOR, trénovali jsme na diff(log(C))
                 # Pokud yw bylo diff(log(C)), pak pred_y je odhad log(C[t0+h]) - log(C[t0+h-1])? NE.
                 # Pokud yw bylo diff(log(C)) zarovnané k t, pak pred_y je log(C[t0+h]) - log(C[t0+h-1])
                 # Pro predikci delty C[t0+h]-C[t0] potřebujeme odhad C[t0+h].
                 # Aproximace: C[t0+h] ≈ C[t0] * exp(pred_y * horizon) ? Nebo jen exp(pred_y)? Záleží na definici yw.
                 # Pokud yw[t] = log(C[t+1])-log(C[t]), pak pred_y je odhad logretu pro t0+1.
                 # Pro horizont H bychom potřebovali sumu logretů.
                 # Zjednodušení: předpokládejme, že pred_y je průměrný logret na bar.
                 pred_future_price = last_price * np.exp(np.clip(pred_y * horizon, -10, 10)) # Extrapolace
                 pred_delta = pred_future_price - last_price
             elif target_type == 'delta':
                  # pred_y je odhad C[t0+h] - C[t0]
                  pred_delta = pred_y
        except OverflowError:
             # print(f"Skipping t0={t0}: Overflow during prediction conversion.")
             continue

        # Skutečná budoucí cena a delta
        future_price = closes[t0 + horizon]
        if np.isnan(future_price):
             # print(f"Skipping t0={t0}: NaN in future_price at t0+h={t0+horizon}.")
             continue
        true_delta = future_price - last_price # Skutečná delta C[t0+h] - C[t0]

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
            "entry_idx": int(t0), # Index času, kdy děláme predikci
            "compare_idx": int(t0 + horizon), # Index času, se kterým srovnáváme
            "last_price": float(last_price), # Cena v čase t0
            "pred_price": float(last_price + pred_delta), # Odhad ceny v t0+horizon
            "future_price": float(future_price), # Skutečná cena v t0+horizon
            "pred_delta": float(pred_delta), # Odhad C[t0+h]-C[t0]
            "true_delta": float(true_delta), # Skutečná C[t0+h]-C[t0]
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
        if n_total < window + horizon + 1:
             print(f"  Reason: Not enough data ({n_total}) for window+horizon+1 ({window + horizon + 1}).")
        else:
             print(f"  Reason: Could be due to NaNs, parameter settings (e.g., min samples), or solver issues.")
        # Vrátíme prázdný dataframe a prázdné summary
        return pd.DataFrame([]), {}


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
    existing_eval_cols = [c for c in wanted_cols if c in df_eval.columns]
    df_eval[existing_eval_cols].to_csv(out_eval_csv, index=False, float_format='%.8f')
    print(f"\nUloženo CSV: {out_eval_csv}")

    # Metriky
    res = metrics(np.array(lasts_price, dtype=float),
                  np.array(preds_delta, dtype=float),
                  np.array(trues_delta, dtype=float))

    # summary JSON
    summary = {
        "symbol": path, "interval": interval, "window": window, "horizon": horizon,
        "baseP": baseP, "sigma": sigma, "N_even": N_even, "N_odd": N_odd,
        "target_type": target_type, "ema_alpha": ema_alpha, "lambda": lam,
        "hit_rate_pred": getattr(res, 'hit_rate_pred', float('nan')),
        "hit_rate_hold": getattr(res, 'hit_rate_hold', float('nan')),
        "corr_pred_true": getattr(res, 'corr_pred_true', float('nan')),
        "mae_price": getattr(res, 'mae_price', float('nan')),
        "mae_return": getattr(res, 'mae_return', float('nan')),
        "count": int(getattr(res, 'count', 0)),
        "corr_shuffle": getattr(res, 'corr_shuffle', float('nan')),
        "corr_lag1": getattr(res, 'corr_lag1', float('nan')),
    }
    out_json = f"sum_h_{base_key}.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    try:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32,
                                      np.float64)):
                    # Nahradí NaN/inf za null
                    return float(obj) if np.isfinite(obj) else None
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        with open(out_json, "w") as f:
            # Převedeme NaN na None pro JSON serializaci
            summary_serializable = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in summary.items()}
            json.dump(summary_serializable, f, indent=2, cls=NpEncoder)
        print(f"Uloženo summary: {out_json}")
    except Exception as e:
        print(f"Error saving JSON summary for {path}: {e}")

    # log výstup do konzole
    print("\n--- HSTRATEGY vs HOLD (Theta Q Basis - FINAL CORRECTED) ---")
    # Použijeme f-string formátování s kontrolou None/NaN
    def fmt(val): return f"{val:9.6f}" if val is not None and np.isfinite(val) else "   nan   "
    print(f"hit_rate_pred:  {fmt(summary['hit_rate_pred'])}")
    print(f"hit_rate_hold:  {fmt(summary['hit_rate_hold'])}")
    print(f"corr_pred_true: {fmt(summary['corr_pred_true'])}")
    print(f"mae_price (delta): {fmt(summary['mae_price'])}")
    print(f"mae_return:     {fmt(summary['mae_return'])}")
    print(f"count:          {summary['count']}\n")
    print(f"Sanity Shuffle Corr: {summary['corr_shuffle']:.4f}")
    print(f"Sanity Lag-1 Corr:   {summary['corr_lag1']:.4f}")

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
            if summary: # Přidáme jen pokud evaluace proběhla a vrátila summary
                rows.append(summary)
        except RuntimeError as re:
             print(f"RUNTIME ERROR processing {sym}: {re}")
             rows.append({"symbol": sym, "error": str(re), **eval_kwargs}) # Přidáme i parametry
        except ValueError as ve:
             print(f"VALUE ERROR processing {sym}: {ve}")
             rows.append({"symbol": sym, "error": str(ve), **eval_kwargs})
        except Exception as e:
            print(f"UNEXPECTED ERROR processing {sym}: {e}")
            rows.append({"symbol": sym, "error": f"Unexpected: {e}", **eval_kwargs})

    df = pd.DataFrame(rows)
    return df

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    """Parsování argumentů příkazové řádky."""
    p = argparse.ArgumentParser(description="Theta Q-Basis Evaluator - FINAL Leak Corrected & Robust.")
    p.add_argument("--symbols", required=True, help="CSV paths, comma-separated")
    p.add_argument("--csv-time-col", default='time', help="Time column name (default: time)")
    p.add_argument("--csv-close-col", default='close', help="Close column name (default: close)")
    # Model Params
    p.add_argument("--window", type=int, default=256, help="Rolling window size.")
    p.add_argument("--horizon", type=int, default=4, help="Prediction horizon.")
    p.add_argument("--baseP", type=float, default=36.0, help="Base period P for z(t)=2pi*t/P.")
    p.add_argument("--sigma", type=float, default=0.8, help="Sigma for q=exp(-pi*sigma).")
    p.add_argument("--N-even", type=int, default=6, help="Max index n for even theta series (cos(2nz)).")
    p.add_argument("--N-odd", type=int, default=6, help="Max index m for odd theta series (cos/sin((2m+1)z)).")
    p.add_argument("--target-type", choices=['logprice', 'logret', 'delta'], default='delta', help="Target variable for regression.")
    p.add_argument("--ema-alpha", type=float, default=0.0, help="Decay factor for time weights (0=uniform, >0 exp decay).")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-3, help="Ridge regularization strength.")
    # Legacy/Unused
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

    # Zajistíme existenci výstupního adresáře předem
    out_path = Path(ARGS.out)
    out_dir = out_path.parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Error creating output directory '{out_dir}': {e}")

    df_summary = run_batch(ARGS.symbols, **eval_kwargs)

    # Příprava finálního výstupního DataFrame
    if not df_summary.empty:
        # Zjistíme, jestli máme nějaké úspěšné řádky
        success_mask = ~df_summary.apply(lambda row: 'error' in row and pd.notna(row['error']), axis=1)
        success_rows = df_summary[success_mask]

        if not success_rows.empty:
            # Sloupce pro výstupní CSV
            cols_out = [
                  "symbol", "target_type", "window", "horizon", "baseP", "sigma",
                  "N_even", "N_odd", "lambda", # Přejmenováno z 'lam' v kódu
                  "hit_rate_pred", "hit_rate_hold", "delta_hit", # delta_hit se dopočítá
                  "corr_pred_true", "mae_price", "mae_return", "count",
                  "corr_shuffle", "corr_lag1" # Přidány sanity check metriky
            ]
            # Přejmenování 'lam' na 'lambda' pro výstup
            if 'lam' in success_rows.columns:
                 # Přejmenujeme jen v úspěšných řádcích pro výstup
                 success_rows = success_rows.rename(columns={"lam": "lambda"})

            # Dopočítání delta_hit
            if 'hit_rate_pred' in success_rows.columns and 'hit_rate_hold' in success_rows.columns:
                 success_rows.loc[:, 'delta_hit'] = success_rows['hit_rate_pred'] - success_rows['hit_rate_hold']
            else:
                 success_rows.loc[:, 'delta_hit'] = np.nan

            # Zajistíme, že všechny sloupce existují, než je vybereme/seřadíme
            existing_cols = [c for c in cols_out if c in success_rows.columns]
            out_df = success_rows[existing_cols] # Použijeme jen úspěšné řádky

            # Uložení a výpis
            try:
                # Ošetření NaN před uložením pro lepší čitelnost
                out_df_save = out_df.fillna('N/A') # Nebo jiný placeholder
                out_df_save.to_csv(out_path, index=False) # Ukládáme bez formátování floatů pro jednoduchost
                print(f"\nUloženo (pouze úspěšné běhy): {out_path}")
                # Výpis s formátováním
                print(out_df.to_string(index=False, float_format='%.6f', na_rep='NaN'))
            except Exception as e:
                print(f"Error saving summary CSV to '{out_path}': {e}")

        # Vypíšeme i chyby, pokud nějaké byly
        error_rows = df_summary[~success_mask]
        if not error_rows.empty:
            print("\nErrors occurred during processing:")
            # Zobrazíme jen symbol a chybu pro přehlednost
            print(error_rows[['symbol', 'error']].to_string(index=False))

    else:
        print("[warn] No records produced (did you pass valid CSV paths or did all runs fail?)")


if __name__ == "__main__":
    main()