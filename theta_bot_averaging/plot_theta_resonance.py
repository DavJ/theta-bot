import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_gpt_ridge_delta/summary_gptRidgeDelta.csv")

# Convert symbol path to short names
df["asset"] = df["symbol"].apply(lambda s: "BTC" if "BTC" in s else "ETH")

plt.figure(figsize=(8,5))
for asset, group in df.groupby("asset"):
    plt.plot(group["horizon"], group["corr_pred_true"], "o-", label=f"{asset} corr(Δ,Δ)")
plt.xlabel("Prediction Horizon (hours)")
plt.ylabel("Correlation Δ_pred vs Δ_true")
plt.title("Theta Resonance Correlation vs Horizon (q=0.5, λ=1e-3)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --- Hit rate comparison ---
plt.figure(figsize=(8,5))
for asset, group in df.groupby("asset"):
    plt.plot(group["horizon"], group["hit_rate_pred"], "o--", label=f"{asset} Hit Rate")
plt.xlabel("Prediction Horizon (hours)")
plt.ylabel("Hit Rate")
plt.title("Theta Resonance Hit Rate vs Horizon")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

