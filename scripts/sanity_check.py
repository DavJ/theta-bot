import sys, os, pandas as pd, numpy as np

if len(sys.argv) < 2:
    print("usage: python scripts/sanity_check.py <csv_path> [time_col=time] [close_col=close]")
    sys.exit(1)

p = sys.argv[1]
tcol = sys.argv[2] if len(sys.argv) > 2 else "time"
ccol = sys.argv[3] if len(sys.argv) > 3 else "close"

if not os.path.exists(p):
    sys.exit(f"[error] CSV not found: {p}")

df = pd.read_csv(p)
if tcol not in df.columns or ccol not in df.columns:
    sys.exit(f"[error] CSV must contain columns '{tcol}' and '{ccol}'")

df = df[[tcol, ccol]].rename(columns={tcol:"time", ccol:"close"})
df["time"]  = pd.to_datetime(df["time"], utc=True, errors="coerce")
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df = df.dropna().sort_values("time").reset_index(drop=True)

print("rows:", len(df), "span:", df.time.min(), "→", df.time.max())
print("dupes:", df.time.duplicated().sum())

dt = df.time.diff().dropna()
print("Δt unique (last 10):")
print(dt.value_counts().sort_index().tail(10))

ret = df["close"].pct_change()
desc = ret.describe(percentiles=[.01,.05,.95,.99])
print("\nreturn stats:\n", desc.to_string())

if desc["std"] > 0.05:
    print("\n[warn] std of returns unusually large -> check price scale (maybe % instead of raw)")
