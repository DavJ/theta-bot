# --- PATCH: ticker support & parent lookup ---
import os, sys, subprocess
from pathlib import Path

def resolve_symbols_to_csv(symbols, interval, limit=400, outdir="prices"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    csvs, tickers = [], []
    for s in symbols:
        s = s.strip()
        if not s:
            continue
        if s.endswith(".csv") and os.path.exists(s):
            csvs.append(s)
        else:
            tickers.append(s)
    if tickers:
        maker = os.environ.get("MAKER_PATH", "make_prices_csv.py")
        if not os.path.exists(maker) and os.path.exists("../make_prices_csv.py"):
            maker = "../make_prices_csv.py"
        if not os.path.exists(maker):
            raise SystemExit("make_prices_csv.py not found – nelze stáhnout CSV pro: " + ",".join(tickers))
        cmd = [sys.executable, maker,
               "--symbols", ",".join(tickers),
               "--interval", interval,
               "--limit", str(limit),
               "--outdir", outdir]
        subprocess.run(cmd, check=True)
        for t in tickers:
            csvs.append(str(Path(outdir) / f"{t}_{interval}.csv"))
    return csvs

def find_eval_with_parent():
    candidates = [
        os.environ.get("EVAL_PATH", ""),
        "theta_eval_hbatch_biquat_max.py",
        "theta_bot_averaging/theta_eval_hbatch_biquat_max.py",
        "../theta_eval_hbatch_biquat_max.py",
        "../theta_bot_averaging/theta_eval_hbatch_biquat_max.py",
    ]
    candidates = [c for c in candidates if c]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise SystemExit("theta_eval_hbatch_biquat_max.py not found. Tried: " + ", ".join(candidates))
# --- PATCH END ---
