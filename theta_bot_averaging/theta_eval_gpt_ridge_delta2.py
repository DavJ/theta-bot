#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ===============================================================
# Jacobi theta-like basis
# ===============================================================
def theta_basis(n_points: int, q: float) -> np.ndarray:
    """
    Generate simple theta-like trigonometric basis functions.
    For now, we just use sine/cosine modulated by q^n.
    """
    x = np.linspace(0, 2 * np.pi, n_points)
    basis = []
    for n in range(1, 17):
        amp = q ** n
        basis.append(amp * np.sin(n * x))
        basis.append(amp * np.cos(n * x))
    return np.array(basis).T


# ===============================================================
# Sliding window generator
# ===============================================================
def make_windows(series: np.ndarray, window: int, horizon: int):
    X, y = [], []
    for i in range(len(series) - window - horizon):
        X.append(series[i : i + window])
        y.append(series[i + window + horizon - 1])  # horizon-ahead target
    return np.array(X), np.array(y)


# ===============================================================
# Walk-forward Ridge Regression
# ===============================================================
def walk_forward_ridge(prices, window, horizon, q, lam, ema_alpha, shuffle_mode=0, diag_out=None):
    print(f"Generating theta basis with q={q} for {window} points...")
    theta = theta_basis(window, q)
    X, y = make_windows(prices, window, horizon)

    if shuffle_mode == 1:
        # Shuffle raw data before windowing
        print("[INFO] Shuffling dataset rows (may break temporal structure)...")
        prices = prices.copy()
        np.random.shuffle(prices)
        X, y = make_windows(prices, window, horizon)

    elif shuffle_mode == 2:
        # Shuffle generated windows
        print("[INFO] Shuffling generated training windows...")
        combined = list(zip(X, y))
        random.shuffle(combined)
        X, y = zip(*combined)
        X, y = np.array(X), np.array(y)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid training samples after shuffle! Try smaller window or disable shuffle.")

    model = Ridge(alpha=lam)
    preds, truths, scores = [], [], []

    for i in range(window, len(X)):
        X_train, y_train = X[:i], y[:i]
        X_test, y_test = X[i : i + 1], y[i : i + 1]

        if len(X_train) == 0 or X_train.ndim < 2:
            print(f"[WARN] Empty or invalid training window at step {i}, skipping...")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]
        preds.append(y_pred)
        truths.append(y_test[0])
        scores.append(r2_score(y[:i], model.predict(X[:i])) if i > 10 else 0)

        if i % 100 == 0:
            print(f"[STEP {i}] R²={scores[-1]:.4f}")

    preds, truths, scores = np.array(preds), np.array(truths), np.array(scores)
    if diag_out:
        pd.DataFrame({
            "pred": preds,
            "true": truths,
            "r2": scores
        }).to_csv(diag_out, index=False)
        print(f"[INFO] Diagnostics saved to {diag_out}")

    return preds, truths, scores


# ===============================================================
# Main
# ===============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True, help="Path to CSV with prices")
    parser.add_argument("--csv-time-col", default="time")
    parser.add_argument("--csv-close-col", default="close")
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--q", type=float, default=0.5, dest="q_param")
    parser.add_argument("--lambda", type=float, default=1e-3, dest="lam")
    parser.add_argument("--ema-alpha", type=float, default=0.0)
    parser.add_argument("--out", default="results.csv")
    parser.add_argument("--diag-out", default=None)
    parser.add_argument("--shuffle", type=int, default=0,
                        help="0 = none, 1 = shuffle rows, 2 = shuffle windows")

    args = parser.parse_args()

    df = pd.read_csv(args.symbols)
    if args.csv_close_col not in df.columns:
        raise ValueError(f"Missing close column '{args.csv_close_col}' in CSV!")

    prices = df[args.csv_close_col].values.astype(float)

    print(f"=== Running {args.symbols} ===")
    preds, truths, scores = walk_forward_ridge(
        prices, args.window, args.horizon, args.q_param,
        args.lam, args.ema_alpha, shuffle_mode=args.shuffle,
        diag_out=args.diag_out
    )

    pd.DataFrame({"pred": preds, "true": truths}).to_csv(args.out, index=False)
    print(f"[DONE] Results saved to {args.out}")
