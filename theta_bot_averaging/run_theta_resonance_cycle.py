import subprocess
import os
import pandas as pd

# === CONFIG ===
symbols = "../prices/BTCUSDT_1h.csv,../prices/ETHUSDT_1h.csv"
out_dir = "results_gpt_ridge_delta"
summary_csv = f"{out_dir}/summary_gptRidgeDelta.csv"
horizons = [6, 8, 10, 12, 16, 20, 24, 32, 40]

base_cmd = [
    "python", "theta_eval_gpt_ridge_delta.py",
    "--symbols", symbols,
    "--csv-time-col", "time",
    "--csv-close-col", "close",
    "--window", "512",
    "--q", "0.5",
    "--lambda", "1e-3",
    "--ema-alpha", "0.0",
    "--out", summary_csv
]

# === Ensure output directory exists ===
os.makedirs(out_dir, exist_ok=True)

# === Run sequentially for each horizon ===
for h in horizons:
    print(f"\n=== Running horizon={h} ===\n")
    cmd = base_cmd + ["--horizon", str(h)]
    subprocess.run(cmd, check=True)

# === Merge results (optional sanity check) ===
df = pd.read_csv(summary_csv)
print("\n✅ Combined results shape:", df.shape)
print(df.groupby("symbol")["horizon"].agg(list))

# === Generate plots ===
subprocess.run(["python", "plot_theta_resonance.py"], check=True)

