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
# Helpery (stejné jako předtím)
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
             print(f"Info: Parsing time column in '{path}' as milliseconds since epoch.")
             time_parsed = pd.to_datetime(df[tcol], unit='ms', utc=True, errors='coerce')
        if time_parsed.isnull().any():
             print(f"Warning: Time parsing failed for some rows in '{path}'. Falling back to row index.")
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

# Funkce ridge zde není potřeba
# def ridge(X, y, lam): ...

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
    corr_pred_true: float
    mae_price: float
    mae_return: float
    count: int
    # Sanity checky zde nejsou relevantní, protože nemáme model učený na Y
    # corr_shuffle: float = float('nan')
    # corr_lag1: float = float('nan')

def metrics(last_prices, pred_delta, true_delta):
    """Vypočítá metriky výkonu."""
    pred_delta = np.asarray(pred_delta).ravel()
    true_delta = np.asarray(true_delta).ravel()
    last_prices = np.asarray(last_prices).ravel()

    valid_mask = ~np.isnan(pred_delta) & ~np.isnan(true_delta) & ~np.isnan(last_prices)
    count = valid_mask.sum()

    if count == 0:
        return EvalResult(float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0)

    pred_d_clean = pred_delta[valid_mask]
    true_d_clean = true_delta[valid_mask]
    last_p_clean = last_prices[valid_mask]

    pred_dir = np.sign(pred_d_clean)
    true_dir = np.sign(true_d_clean)
    mask_nonzero_true = (true_dir != 0)
    hit_pred = np.mean(pred_dir[mask_nonzero_true] == true_dir[mask_nonzero_true]) if mask_nonzero_true.sum() > 0 else float('nan')

    hold_up = (true_d_clean > 0).astype(int)
    hit_hold = np.mean(hold_up) if len(hold_up) > 0 else float('nan')

    c = _corr(pred_d_clean, true_d_clean)
    mae_p = np.mean(np.abs(true_d_clean - pred_d_clean))

    valid_ret_mask = (np.abs(last_p_clean) > 1e-9)
    mae_r = float('nan')
    if valid_ret_mask.sum() > 0:
        lp_valid = last_p_clean[valid_ret_mask]
        pred_d_valid = pred_d_clean[valid_ret_mask]
        true_d_valid = true_d_clean[valid_ret_mask]
        pred_ret = np.divide(pred_d_valid, lp_valid, out=np.zeros_like(pred_d_valid), where=lp_valid!=0)
        true_ret = np.divide(true_d_valid, lp_valid, out=np.zeros_like(true_d_valid), where=lp_valid!=0)
        mae_r = np.mean(np.abs(true_ret - pred_ret))

    return EvalResult(hit_pred, hit_hold, c, mae_p, mae_r, count)

def is_csv_symbol(s):
    return isinstance(s, str) and s.strip().lower().endswith('.csv')

def gaussian_weights(n, sigma_factor):
    """Gaussovy váhy centrované na konec okna."""
    if sigma_factor <= 0: return np.ones(n) / n # Uniformní, pokud sigma <= 0
    # sigma = sigma_factor * n # Lineární škálování sigma s délkou okna
    # Robustnější: sigma jako pevný počet barů, např. sigma_factor = 50 barů
    sigma = max(1.0, sigma_factor) # Použijeme sigma_factor přímo jako std dev v barech
    indices = np.arange(n)
    # Váha klesá s vzdáleností od posledního bodu (n-1)
    weights = np.exp(-0.5 * ((indices - (n - 1)) / sigma) ** 2)
    return weights / (weights.sum() + 1e-12)


# -----------------------------
# Theta Báze (stejná jako předtím)
# -----------------------------

def build_theta_q_basis(t_idx: np.ndarray, baseP: float, sigma: float, N_even: int, N_odd: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vytvoří základní (neortogonalizovanou) bázi z q-řad Jacobiho thet."""
    t = np.asarray(t_idx, dtype=float)
    N = len(t)
    sigma = max(sigma, 1e-9)
    q = np.exp(-np.pi * sigma)
    baseP = max(baseP, 1e-9)
    omega = 2.0 * np.pi / baseP
    z = omega * t
    cols = []
    q_weights = []
    cutoff_thresh = 1e-9

    for k in range(1, N_even + 1):
        q_pow_k2 = np.abs(q)**(k**2)
        if q_pow_k2 < cutoff_thresh: break
        term_cos = np.cos(2 * k * z)
        cols.append(term_cos)
        q_weights.append(2 * q_pow_k2)
        cols.append((-1)**k * term_cos)
        q_weights.append(2 * q_pow_k2)

    for m in range(N_odd):
        idx_float = m + 0.5
        q_pow_m2 = np.abs(q)**(idx_float**2)
        if q_pow_m2 < cutoff_thresh: break
        angle = (2 * m + 1) * z
        cols.append(np.cos(angle))
        q_weights.append(2 * q_pow_m2)
        cols.append((-1)**m * np.sin(angle))
        q_weights.append(2 * q_pow_m2)

    if not cols:
        print("Warning: No basis columns generated.")
        return np.zeros((N, 0)), np.zeros(0)

    B_raw = np.stack(cols, axis=1)
    q_weights = np.array(q_weights, dtype=float)
    q_weights = np.maximum(q_weights, 1e-12)
    return B_raw, q_weights

# -----------------------------
# Jádro evaluace (ČISTÁ EXTRAPOLACE)
# -----------------------------

def evaluate_symbol_csv(path: str,
                        interval: str,
                        window: int,
                        horizon: int,
                        baseP: float,
                        sigma: float, # Sigma pro q-váhy
                        N_even: int,
                        N_odd: int,
                        gauss_sigma: float, # Sigma pro Gaussovo okno projekce
                        # target_type, lam, pred_ensemble, max_by - NEPOTŘEBNÉ
                        ):

    df_in = _read_price_csv(path)
    times = df_in["time"].to_numpy()
    closes = df_in["close"].to_numpy().astype(float)
    n_total = len(closes)

    # 1. Výpočet celé theta báze B_all_raw
    t_idx = np.arange(n_total, dtype=float)
    B_all_raw, q_weights = build_theta_q_basis(t_idx, baseP, sigma, N_even, N_odd)
    D = B_all_raw.shape[1]
    q_weights_sqrt = np.sqrt(q_weights) if D > 0 else np.array([])

    rows = []
    preds_delta, trues_delta, lasts_price = [], [], []

    # 2. Výpočet "současných koeficientů" alpha pro každý čas t0
    # Použijeme metodu podobnou _build_features z ...bak4-gpt
    Alpha_all = np.zeros((n_total, D), dtype=float)
    if D > 0:
        # Gaussovy váhy pro projekci tvaru ceny
        gauss_w = gaussian_weights(window, gauss_sigma)

        for t0 in range(window - 1, n_total):
            # Segment ceny končící v t0
            seg = closes[t0 - window + 1 : t0 + 1]
            if len(seg) != window or np.isnan(seg).any():
                continue # Přeskočíme, pokud okno není plné nebo obsahuje NaN

            # z-normalizace okna
            seg_mean = np.mean(seg)
            seg_std = np.std(seg)
            if seg_std < 1e-9: continue # Přeskočíme konstantní okno
            seg_norm = (seg - seg_mean) / seg_std

            # Theta báze pro toto okno (jen časové indexy 0..window-1)
            t_win_idx = np.arange(window, dtype=float)
            # Potřebujeme B_window_raw, NE B_all_raw[t0-window+1:t0+1]
            # Musíme přepočítat bázi relativně k oknu
            B_win_raw_rel, _ = build_theta_q_basis(t_win_idx, baseP, sigma, N_even, N_odd)
            if B_win_raw_rel.shape[1] != D: continue # Kontrola konzistence

            # Projekce normalizovaného tvaru ceny na bázi okna s Gaussovým vážením
            # alpha = suma( segment * báze * váha )
            # Vážený segment
            seg_norm_weighted = seg_norm * gauss_w
            # Projekce = součin váženého segmentu s bázovými vektory
            alpha_t0 = seg_norm_weighted @ B_win_raw_rel # Výsledkem je vektor [D]
            Alpha_all[t0, :] = alpha_t0

    # 3. Walk-forward predikce
    start_idx = window # První index t0, kde máme alpha a můžeme predikovat
    if start_idx >= n_total - horizon:
         needed = window + horizon
         raise RuntimeError(f"Nedostatek dat ({n_total}) pro window={window} a horizon={horizon}. Potřeba alespoň {needed}.")

    print(f"Starting pure extrapolation from index t0={start_idx} up to {n_total - horizon - 1}")

    for t0 in range(start_idx, n_total - horizon):

        # --- ČISTÁ EXTRAPOLACE ---
        # 1. Získáme "současné koeficienty" (theta spektrum tvaru ceny) z času t0
        alpha_now = Alpha_all[t0, :]
        if np.isnan(alpha_now).any() or D==0: # Pokud alpha nebylo spočítáno nebo D=0
             continue

        # 2. Získáme vektor BUDOUCÍ theta báze
        t_future = t0 + horizon
        b_raw_future = B_all_raw[t_future, :]
        if np.isnan(b_raw_future).any():
             continue

        # 3. Predikce = suma( alpha_současné * b_budoucí )
        # Možná bychom měli vážit q-váhami? Zkusme bez nich i s nimi.
        # Varianta A: Bez q-vah
        # pred_y_raw = float( alpha_now @ b_raw_future )
        # Varianta B: S q-váhami (aplikujeme na alpha i na bázi?)
        # alpha vážené? q_weights_sqrt zde není správně, alpha už je "vážené" projekcí.
        # Spíše vážíme budoucí bázi:
        pred_y_weighted = float( alpha_now @ (b_raw_future * q_weights_sqrt) )
        pred_y = pred_y_weighted # Vybereme variantu B

        # --- Převod predikce y na predikci delty ---
        # TENTO KROK JE PROBLEMATICKÝ - co vlastně pred_y reprezentuje?
        # alpha jsou projekce NORMALIZOVANÉ ceny. pred_y je tedy v jednotkách std. dev.
        # Potřebujeme převést zpět na cenovou škálu.
        # Získáme std. dev. z okna končícího v t0
        seg_t0 = closes[t0 - window + 1 : t0 + 1]
        if len(seg_t0) != window or np.isnan(seg_t0).any(): continue
        seg_t0_std = np.std(seg_t0)
        if seg_t0_std < 1e-9: continue

        # Predikovaná změna v jednotkách std. dev.
        pred_delta_norm = pred_y
        # Predikovaná změna v cenových jednotkách
        pred_delta = pred_delta_norm * seg_t0_std

        # Skutečná budoucí cena a delta
        last_price = closes[t0]
        future_price = closes[t0 + horizon]
        if np.isnan(last_price) or np.isnan(future_price): continue
        true_delta = future_price - last_price

        # Finální kontrola platnosti
        if np.isnan(pred_delta) or np.isnan(true_delta) or not np.isfinite(pred_delta) or not np.isfinite(true_delta):
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
        print(f"Warning: No valid prediction rows produced for {path}.")
        # ... (zbytek chybové hlášky) ...
        return pd.DataFrame([]), {}


    df_eval = pd.DataFrame(rows)

    # --- Ukládání a metriky ---
    base_key = _safe_stem_key(path)
    out_eval_csv = f"eval_h_{base_key}_purextr.csv" # Jiný název souboru
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
        "gauss_sigma": gauss_sigma, # Přidán parametr
        # Odebrány parametry regrese
        "hit_rate_pred": getattr(res, 'hit_rate_pred', float('nan')),
        "hit_rate_hold": getattr(res, 'hit_rate_hold', float('nan')),
        "corr_pred_true": getattr(res, 'corr_pred_true', float('nan')),
        "mae_price": getattr(res, 'mae_price', float('nan')),
        "mae_return": getattr(res, 'mae_return', float('nan')),
        "count": int(getattr(res, 'count', 0)),
    }
    out_json = f"sum_h_{base_key}_purextr.json" # Jiný název souboru
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
    print("\n--- HSTRATEGY vs HOLD (Pure Theta Extrapolation) ---")
    def fmt(val): return f"{val:9.6f}" if val is not None and np.isfinite(val) else "   nan   "
    print(f"hit_rate_pred:  {fmt(summary['hit_rate_pred'])}")
    print(f"hit_rate_hold:  {fmt(summary['hit_rate_hold'])}")
    print(f"corr_pred_true: {fmt(summary['corr_pred_true'])}")
    print(f"mae_price (delta): {fmt(summary['mae_price'])}")
    print(f"mae_return:     {fmt(summary['mae_return'])}")
    print(f"count:          {summary['count']}\n")

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
# CLI (Upraveno pro Pure Extrapolation)
# -----------------------------

def parse_args():
    """Parsování argumentů příkazové řádky."""
    p = argparse.ArgumentParser(description="Pure Theta Q-Basis Extrapolation Evaluator.")
    p.add_argument("--symbols", required=True, help="CSV paths, comma-separated")
    p.add_argument("--csv-time-col", default='time', help="Time column name (default: time)")
    p.add_argument("--csv-close-col", default='close', help="Close column name (default: close)")
    # Model Params
    p.add_argument("--window", type=int, default=256, help="Window size for shape projection.")
    p.add_argument("--horizon", type=int, default=4, help="Prediction horizon.")
    p.add_argument("--baseP", type=float, default=36.0, help="Base period P for z(t)=2pi*t/P.")
    p.add_argument("--sigma", type=float, default=0.8, help="Sigma for q=exp(-pi*sigma) in basis.")
    p.add_argument("--N-even", type=int, default=6)
    p.add_argument("--N-odd", type=int, default=6)
    p.add_argument("--gauss-sigma", type=float, default=50.0, help="Sigma (in bars) for Gaussian window in projection.")
    # Odebrány parametry pro regresi: --lambda, --target-type, --ema-alpha
    p.add_argument("--out", required=True, help="Summary CSV output path.")
    # Ponechány pro kompatibilitu, ale nepoužity
    p.add_argument("--interval", default="1h")
    p.add_argument("--phase", default="theta_pure")
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
        "baseP": ARGS.baseP,
        "sigma": ARGS.sigma,
        "N_even": ARGS.N_even,
        "N_odd": ARGS.N_odd,
        "gauss_sigma": ARGS.gauss_sigma,
        # Tyto parametry zde nejsou potřeba, ale můžeme je předat pro info v summary
        # "target_type": "N/A",
        # "lam": "N/A",
        # "pred_ensemble": "N/A",
        # "max_by": "N/A",
    }

    out_path = Path(ARGS.out)
    out_dir = out_path.parent
    try: out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e: print(f"Warning: Error creating output directory '{out_dir}': {e}")

    df_summary = run_batch(ARGS.symbols, **eval_kwargs)

    if not df_summary.empty:
        success_mask = ~df_summary.apply(lambda row: 'error' in row and pd.notna(row['error']), axis=1)
        success_rows = df_summary[success_mask]

        if not success_rows.empty:
            # Sloupce pro výstupní CSV (upravené)
            cols_out = [
                  "symbol", "window", "horizon", "baseP", "sigma", "gauss_sigma",
                  "N_even", "N_odd",
                  "hit_rate_pred", "hit_rate_hold", "delta_hit",
                  "corr_pred_true", "mae_price", "mae_return", "count"
            ]
            if 'hit_rate_pred' in success_rows.columns and 'hit_rate_hold' in success_rows.columns:
                 success_rows.loc[:, 'delta_hit'] = success_rows['hit_rate_pred'] - success_rows['hit_rate_hold']
            else:
                 success_rows.loc[:, 'delta_hit'] = np.nan

            existing_cols = [c for c in cols_out if c in success_rows.columns]
            out_df = success_rows[existing_cols]

            try:
                out_df_save = out_df.fillna('N/A')
                out_df_save.to_csv(out_path, index=False)
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