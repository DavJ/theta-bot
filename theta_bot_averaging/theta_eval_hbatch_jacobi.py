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
# Helpery (Ridge, Fetch CSV, Metrics zůstávají stejné)
# -----------------------------

def ridge(X, y, lam):
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

def fetch_csv(path, time_col, close_col):
    df = pd.read_csv(path)
    if time_col not in df.columns or close_col not in df.columns:
        raise ValueError(f"CSV '{path}' must contain columns '{time_col}' and '{close_col}'. Columns: {list(df.columns)}")
    out = df[[time_col, close_col]].rename(columns={time_col:'time', close_col:'close'}).copy()
    # Robustnější parsování času
    try:
        # Zkusíme s jednotkami (ms, s) pokud je numerický
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

@dataclass
class EvalResult:
    hit_rate_pred: float
    hit_rate_hold: float
    corr_pred_true: float
    mae_price: float  # Změněno na MAE predikované delty
    mae_return: float # Změněno na MAE predikované návratnosti
    count: int

def metrics(last_prices, pred_delta, true_delta):
    if len(true_delta) == 0:
        return EvalResult(float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 0)

    pred_dir = np.sign(pred_delta)
    true_dir = np.sign(true_delta)
    # Ignorujeme případy, kdy true_dir je 0 pro hit rate
    mask_nonzero_true = (true_dir != 0)
    if mask_nonzero_true.sum() > 0:
         hit_pred = (pred_dir[mask_nonzero_true] == true_dir[mask_nonzero_true]).mean()
    else:
         hit_pred = float('nan')

    # Hold benchmark: predikuje vždy růst (nebo směr předchozího baru - zde zjednodušeno)
    hold_up = (true_delta > 0).astype(int)
    hit_hold = hold_up.mean() if len(hold_up) > 0 else float('nan')

    # Korelace
    c = np.corrcoef(pred_delta, true_delta)[0,1] if len(true_delta) > 1 else float('nan')

    # MAE
    mae_p = np.mean(np.abs(true_delta - pred_delta))
    # MAE návratnosti (predikovaná vs skutečná)
    pred_ret = pred_delta / (last_prices + 1e-12)
    true_ret = true_delta / (last_prices + 1e-12)
    mae_r = np.mean(np.abs(true_ret - pred_ret))

    return EvalResult(hit_pred, hit_hold, c, mae_p, mae_r, len(true_delta))

def is_csv_symbol(s):
    return s.lower().endswith('.csv')

def _safe_stem_key(path: str) -> str:
    # pro jména eval souborů jako "eval_h_BTCUSDT_1HCSV.csv"
    base = Path(path).name
    stem = Path(path).stem
    # Odstraníme tečky a převedeme na velká písmena pro konzistenci
    return stem.replace(".", "").upper()


# -----------------------------
# Nová Theta Báze podle paperu
# -----------------------------

def build_theta_q_basis(t_idx: np.ndarray, baseP: float, sigma: float, N_even: int, N_odd: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vytvoří základní (neortogonalizovanou) bázi z q-řad Jacobiho thet.
    Používá z(t) = omega * t.
    Vrací: Matice báze B_raw [N, D], Vektor q-vah [D]
    """
    t = np.asarray(t_idx, dtype=float)
    N = len(t)
    q = np.exp(-np.pi * sigma)
    omega = 2.0 * np.pi / baseP
    z = omega * t

    cols = []
    q_weights = []

    # Theta 3 & Theta 0 (sudé cos)
    # Přidáme konstantu pro k=0 (theta3 dává 1)
    # cols.append(np.ones(N)) # Volitelné - může být absorbováno regularizací
    # q_weights.append(1.0)
    for k in range(1, N_even + 1):
        q_pow_k2 = np.abs(q)**(k**2)
        # Theta 3
        cols.append(np.cos(2 * k * z))
        q_weights.append(2 * q_pow_k2)
        # Theta 0
        cols.append((-1)**k * np.cos(2 * k * z))
        q_weights.append(2 * q_pow_k2)

    # Theta 2 (cos) & Theta 1 (sin) (liché)
    for m in range(N_odd): # N_odd je počet termů, tedy m jde od 0 do N_odd-1
        idx_float = m + 0.5
        q_pow_m2 = np.abs(q)**(idx_float**2)
        angle = (2 * m + 1) * z
        # Theta 2
        cols.append(np.cos(angle))
        q_weights.append(2 * q_pow_m2)
        # Theta 1
        cols.append((-1)**m * np.sin(angle))
        q_weights.append(2 * q_pow_m2)

    if not cols:
        return np.zeros((N, 0)), np.zeros(0)

    B_raw = np.stack(cols, axis=1) # Shape (N, D)
    q_weights = np.array(q_weights, dtype=float)

    # Ošetření velmi malých vah
    q_weights = np.maximum(q_weights, 1e-12)

    return B_raw, q_weights


def weighted_qr_simple(B: np.ndarray, q_weights: np.ndarray, time_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zjednodušená vážená QR dekompozice.
    Váhuje řádky (čas) a sloupce (q-váhy).
    Vrací: Orthonormální bázi Q, Horní trojúhelníkovou matici R (pro info)
    """
    W, D = B.shape
    if D == 0:
        return np.zeros((W, 0)), np.zeros((0, 0))

    # Váhování sloupců (q-váhy)
    B_weighted_cols = B * np.sqrt(q_weights)[None, :]

    # Váhování řádků (časové váhy, např. Gauss nebo EMA)
    B_weighted_rows_cols = B_weighted_cols * np.sqrt(time_weights)[:, None]

    # Standardní QR na plně vážené matici
    try:
         Q, R = np.linalg.qr(B_weighted_rows_cols)
         # Q má nyní ortonormální sloupce vzhledem k váhovanému prostoru
         # Normalizace R pro konzistenci (volitelné)
         # R = R / np.sqrt(q_weights)[None, :] / np.sqrt(time_weights)[:,None] # Toto je složité
    except np.linalg.LinAlgError:
         print("Warning: QR decomposition failed, returning zeros.")
         Q = np.zeros_like(B)
         R = np.zeros((D,D))

    # Q je nyní "ortonormální" báze v transformovaném prostoru
    return Q, R

# -----------------------------
# Jádro evaluace (podle paperu)
# -----------------------------

def evaluate_symbol_csv(path: str,
                        interval: str, # Nyní nevyužito, ale ponecháno pro API
                        window: int,
                        horizon: int,
                        # Parametry pro theta bázi
                        baseP: float,
                        sigma: float,
                        N_even: int,
                        N_odd: int,
                        # Ostatní
                        target_type: str, # 'logprice', 'logret', 'delta'
                        ema_alpha: float, # Pro časové váhy ortogonalizace (0=žádné, 1=poslední bod)
                        lam: float, # Ridge regularizace
                        pred_ensemble: str, # Nyní asi jen 'avg' má smysl
                        max_by: str): # Nyní nevyužito

    df_in = fetch_csv(path, ARGS.csv_time_col or 'time', ARGS.csv_close_col or 'close')
    times = df_in["time"].to_numpy()
    closes = df_in["close"].to_numpy().astype(float)

    # 1. Příprava cílové veličiny (y_target)
    if target_type == 'logprice':
        y_target = np.log(closes + 1e-9) # Log ceny
    elif target_type == 'logret':
        y_target = np.log(closes + 1e-9)
        y_target = np.diff(y_target, prepend=y_target[0]) # Log návratnost
    elif target_type == 'delta':
         # Predikujeme budoucí změnu ceny
         # Musíme y_target posunout tak, aby y_target[t] = closes[t+horizon] - closes[t]
         y_target = np.zeros_like(closes)
         y_target[:-horizon] = closes[horizon:] - closes[:-horizon]
         # Posledních 'horizon' hodnot bude NaN nebo 0, což je v pořádku pro trénink
    else:
        raise ValueError(f"Neznámý target_type: {target_type}")

    # 2. Výpočet celé (nevážené) theta báze pro všechna data
    t_idx = np.arange(len(closes), dtype=float)
    B_all_raw, q_weights = build_theta_q_basis(t_idx, baseP, sigma, N_even, N_odd)
    D = B_all_raw.shape[1] # Dimenze báze

    rows = []
    preds_delta, trues_delta, lasts_price = [], [], []

    # 3. Walk-forward evaluace
    # Začínáme až máme dost historie pro okno A dost budoucnosti pro cíl
    start_idx = window
    if start_idx >= len(closes) - horizon:
         raise RuntimeError(f"Nedostatek dat ({len(closes)}) pro window={window} a horizon={horizon}.")

    for t0 in range(start_idx, len(closes) - horizon):
        # Indexy pro trénovací okno: [t0 - window, t0 - 1]
        lo = t0 - window
        hi = t0

        # --- Ortonormalizace báze na okně ---
        B_win_raw = B_all_raw[lo:hi, :]
        # Časové váhy (např. EMA nebo Gauss)
        time_weights = np.exp(ema_alpha * (np.arange(window) - (window - 1)))
        time_weights /= time_weights.sum()

        Q_win, R_win = weighted_qr_simple(B_win_raw, q_weights, time_weights)
        # Q_win je nyní ortonormální báze [W, D] pro toto okno

        # --- Projekce signálu na bázi ---
        y_win = y_target[lo:hi]

        # Ridge regrese na ortonormální bázi
        # Pozn.: Pro dokonale ortonormální bázi (Q^T Q = I) by ridge byl jednodušší,
        # ale zde ponecháme obecný vzorec pro robustnost.
        beta_hat = ridge(Q_win, y_win, lam) # Koeficienty [D]

        # --- Extrapolace báze a Predikce ---
        # Musíme spočítat vektor ORTONORMÁLNÍ báze v budoucím čase t0 + horizon
        # Toto je zjednodušení: použijeme RAW bázi v budoucnu a RAW koeficienty.
        # Správně bychom měli aplikovat transformaci z QR.

        # Zjednodušený přístup: Trénujeme na RAW bázi s váhami
        # Aplikujeme časové váhy přímo na data pro ridge
        B_win_raw_weighted_rows = B_win_raw * np.sqrt(time_weights)[:, None]
        y_win_weighted_rows = y_win * np.sqrt(time_weights)
        # Aplikujeme q-váhy na sloupce pro ridge
        B_win_fully_weighted = B_win_raw_weighted_rows * np.sqrt(q_weights)[None, :]

        # Ridge na vážené RAW bázi
        beta_raw_weighted = ridge(B_win_fully_weighted, y_win_weighted_rows, lam)

        # Predikce pomocí RAW báze v čase t0 (pro predikci t0+horizon)
        # Používáme vektor báze z času t0-1 (poslední známý) pro predikci t0+horizon
        b_raw_now = B_all_raw[t0 - 1, :]

        # Predikce: SUMA(beta_j * b_j(t0+h)) - zde aproximováno
        # Používáme beta z vážené regrese na raw bázi
        # Musíme škálovat beta zpět nebo váhovat b_raw_now
        pred_y = float( (b_raw_now * np.sqrt(q_weights)) @ beta_raw_weighted )

        # --- Převod predikce y na predikci delty ---
        last_price = closes[t0 - 1] # Cena v čase t0-1

        if target_type == 'logprice':
            pred_future_logprice = pred_y
            pred_future_price = np.exp(pred_future_logprice)
            pred_delta = pred_future_price - last_price
        elif target_type == 'logret':
            pred_logret = pred_y
            # Odhad budoucí ceny: C(t+h) = C(t) * exp(logret(t+h))
            # Zde predikujeme logret pro okno [t, t+h], použijeme last_price v t-1
            pred_future_price = last_price * np.exp(pred_logret) # Aproximace
            pred_delta = pred_future_price - last_price
        elif target_type == 'delta':
             pred_delta = pred_y # Přímo predikujeme deltu
        else: # default delta
             pred_delta = pred_y

        # Skutečná budoucí cena a delta pro srovnání
        future_price = closes[t0 - 1 + horizon]
        true_delta = future_price - last_price

        # Uložení výsledků
        pred_dir = int(np.sign(pred_delta))
        true_dir = int(np.sign(true_delta))
        correct_pred_val = 1 if pred_dir != 0 and pred_dir == true_dir else 0

        rows.append({
            "time": str(times[t0 - 1]),
            "entry_idx": int(t0 - 1),
            "compare_idx": int(t0 - 1 + horizon),
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
    # Zajistíme existenci adresáře
    Path(out_eval_csv).parent.mkdir(parents=True, exist_ok=True)

    wanted_cols = [
        "time","entry_idx","compare_idx",
        "last_price","pred_price","future_price",
        "pred_delta","true_delta","pred_dir","true_dir","correct_pred"
    ]
    df_eval[wanted_cols].to_csv(out_eval_csv, index=False)
    print(f"\nUloženo CSV: {out_eval_csv}")

    # Metriky
    res = metrics(np.array(lasts_price), np.array(preds_delta), np.array(trues_delta))

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
    # Zajistíme existenci adresáře
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Uloženo summary: {out_json}")
    except Exception as e:
        print(f"Error saving JSON summary: {e}")


    # log výstup do konzole
    print("\n--- HSTRATEGY vs HOLD (Theta Q Basis) ---")
    print(f"hit_rate_pred:  {res.hit_rate_pred:9.6f}")
    print(f"hit_rate_hold:  {res.hit_rate_hold:9.6f}")
    print(f"corr_pred_true: {res.corr_pred_true:9.6f}")
    print(f"mae_price (delta): {res.mae_price:9.6f}") # MAE na deltách
    print(f"mae_return:     {res.mae_return:9.6f}") # MAE na návratnostech
    print(f"count:          {res.count}\n")

    return df_eval, summary


def run_batch(symbol_paths: str, **eval_kwargs):
    syms = [s.strip() for s in symbol_paths.split(",") if s.strip()]
    rows = []
    for sym in syms:
        print(f"\n=== Running {sym} ===\n")
        try:
            df_eval, summary = evaluate_symbol_csv(sym, **eval_kwargs)
            rows.append(summary)
        except Exception as e:
            print(f"ERROR processing {sym}: {e}")
            # Můžeme přidat záznam o chybě, pokud chceme
            rows.append({"symbol": sym, "error": str(e)})


    df = pd.DataFrame(rows)
    return df

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Theta Q-Basis Evaluator based on LaTeX specs.")
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
    # Legacy/Unused (ale pro konzistenci s voláním)
    p.add_argument("--pred-ensemble", default="avg")
    p.add_argument("--max-by", default="transform")
    p.add_argument("--interval", default="1h") # Nyní nevyužito
    p.add_argument("--phase", default="theta_q") # Pro info

    p.add_argument("--out", required=True, help="Summary CSV output path.")
    return p.parse_args()


def main():
    global ARGS # Abychom měli přístup v evaluate_symbol_csv
    ARGS = parse_args()

    # Předáme relevantní argumenty do run_batch
    eval_kwargs = {
        "interval": ARGS.interval, # Ponecháno pro summary
        "window": ARGS.window,
        "horizon": ARGS.horizon,
        "baseP": ARGS.baseP,
        "sigma": ARGS.sigma,
        "N_even": ARGS.N_even,
        "N_odd": ARGS.N_odd,
        "target_type": ARGS.target_type,
        "ema_alpha": ARGS.ema_alpha,
        "lam": ARGS.lam,
        "pred_ensemble": ARGS.pred_ensemble, # Pro summary
        "max_by": ARGS.max_by, # Pro summary
    }

    df_summary = run_batch(ARGS.symbols, **eval_kwargs)

    # výstupní tabulka pro souhrnné CSV
    if not df_summary.empty and 'error' not in df_summary.columns:
         cols_out = [
              "symbol", "target_type", "window", "horizon", "baseP", "sigma", "N_even", "N_odd", "lambda",
              "hit_rate_pred", "hit_rate_hold", "corr_pred_true", "mae_price", "mae_return", "count"
         ]
         # Přidáme delta_hit pro srovnání
         if 'hit_rate_pred' in df_summary.columns and 'hit_rate_hold' in df_summary.columns:
              df_summary['delta_hit'] = df_summary['hit_rate_pred'] - df_summary['hit_rate_hold']
              cols_out.insert(11, 'delta_hit') # Vložíme za hit_rate_hold

         # Přejmenování sloupců pro finální report, pokud je potřeba
         # df_summary = df_summary.rename(columns={"lambda": "ridge_lambda"})

         # Vybereme a seřadíme sloupce
         out_df = df_summary.reindex(columns=cols_out, fill_value=np.nan)

         out_path = ARGS.out
         out_dir = Path(out_path).parent
         out_dir.mkdir(parents=True, exist_ok=True)
         out_df.to_csv(out_path, index=False, float_format='%.6f')
         print(f"\nUloženo: {out_path}")
         print(out_df.to_string(index=False, float_format='%.6f'))
    elif not df_summary.empty:
         print("\nErrors occurred during processing:")
         print(df_summary[df_summary['error'].notna()].to_string(index=False))
    else:
        print("[warn] No records written to summary (did you pass valid CSV paths?)")


if __name__ == "__main__":
    main()