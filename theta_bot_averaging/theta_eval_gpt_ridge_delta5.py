import argparse
import numpy as np
import pandas as pd
import os
from datetime import datetime, timezone
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# === Theta Utility Functions ===

def load_data(path, time_col, close_col):
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    df = df[[time_col, close_col]].dropna()
    return df

def make_features(series, window=24):
    """Create lag-based features for regression"""
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# === Core Walk-Forward Evaluation ===

def walk_forward_theta(prices, alpha=1.0, window=24, step=1):
    scaler = StandardScaler()
    preds, reals = [], []

    X, y = make_features(prices, window=window)
    for i in range(1, len(X) - step):
        X_train, y_train = X[:i], y[:i]
        X_test = X[i].reshape(1, -1)
        y_test = y[i]

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)[0]

        preds.append(pred)
        reals.append(y_test)

    return np.array(preds), np.array(reals)


# === Statistical Tests ===

def random_phase_test(prices, repeats=50):
    results = []
    for _ in tqdm(range(repeats), desc="Phase rotations"):
        shifted = np.roll(prices, np.random.randint(1, len(prices)))
        preds, reals = walk_forward_theta(shifted)
        score = -mean_squared_error(reals, preds)
        results.append(score)
    return summary_stats(results, "Random Phase")

def permutation_batch_test(prices, repeats=50):
    results = []
    for _ in tqdm(range(repeats), desc="Permutations"):
        shuffled = np.random.permutation(prices)
        preds, reals = walk_forward_theta(shuffled)
        score = -mean_squared_error(reals, preds)
        results.append(score)
    return summary_stats(results, "Permutation")

def synthetic_noise_test(prices, repeats=50, noise_level=0.02):
    results = []
    for _ in tqdm(range(repeats), desc="Noise runs"):
        noisy = prices + np.random.normal(0, noise_level * np.std(prices), len(prices))
        preds, reals = walk_forward_theta(noisy)
        score = -mean_squared_error(reals, preds)
        results.append(score)
    return summary_stats(results, "Noise")

def summary_stats(results, label):
    arr = np.array(results)
    stats = {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
    }
    print(f"✅ {label} hotovo: {stats}")
    return stats

# === Main Routine ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--csv-time-col", default="time")
    parser.add_argument("--csv-close-col", default="close")
    parser.add_argument("--outdir", default="experiments_delta4/")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.symbols, args.csv_time_col, args.csv_close_col)
    prices = df[args.csv_close_col].values

    print("\n🧠 Spouštím hlavní Walk-Forward Theta Ridge test...")
    preds, reals = walk_forward_theta(prices)
    base_score = -mean_squared_error(reals, preds)
    print(f"✅ Základní skóre (Ridge): {base_score:.6f}")

    print("\n🌀 Spouštím Random Phase Test...")
    random_phase = random_phase_test(prices, repeats=args.repeats)

    print("\n🔀 Spouštím Permutation Batch Test...")
    permutation = permutation_batch_test(prices, repeats=args.repeats)

    print("\n📉 Spouštím Synthetic Noise Test...")
    noise = synthetic_noise_test(prices, repeats=args.repeats)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_score": base_score,
        "random_phase": random_phase,
        "permutation": permutation,
        "noise": noise,
    }

    outpath = os.path.join(args.outdir, "theta_eval_summary.json")
    pd.Series(summary).to_json(outpath, indent=2)
    print(f"\n🧭 Kompletní výsledky uložené do: {outpath}")

if __name__ == "__main__":
    main()

