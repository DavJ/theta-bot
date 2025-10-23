#!/usr/bin/env python
import argparse, os, subprocess, sys, shutil

def ensure_csv(symbol, interval):
    # if CSV exists, return it; else try to fetch using local make_prices_csv.py
    csv_path = os.path.join("prices", f"{symbol.upper()}_{interval}.csv")
    if os.path.exists(csv_path):
        return csv_path
    fetcher = "make_prices_csv.py"
    if not os.path.exists(fetcher):
        print("make_prices_csv.py nebyl nalezen, nelze stáhnout ceny.", file=sys.stderr)
        sys.exit(1)
    os.makedirs("prices", exist_ok=True)
    cmd = [sys.executable, fetcher, "--symbols", symbol.upper(), "--interval", interval, "--limit", "1000"]
    subprocess.run(cmd, check=True)
    if not os.path.exists(csv_path):
        print(f"Po stažení soubor {csv_path} stále neexistuje.", file=sys.stderr)
        sys.exit(1)
    return csv_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--interval", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    csv_path = ensure_csv(args.symbol, args.interval)

    # run evaluator to produce per-bar eval csv
    cmd = [
        sys.executable, "theta_bot_averaging/theta_eval_hbatch_biquat_max.py",
        "--symbols", csv_path,
        "--interval", args.interval, "--window", "256", "--horizon", "4",
        "--minP", "24", "--maxP", "480", "--nP", "16",
        "--sigma", "0.8", "--lambda", "1e-3",
        "--pred-ensemble", "avg", "--max-by", "transform",
        "--out", "results/hbatch_biquat_summary_live.csv"
    ]
    subprocess.run(cmd, check=True)

    print(f"=== Running {csv_path} ===")
    print("Hotovo. Per-bar predikce jsou v eval_h_<BASENAME>.csv")
    if not args.dry_run:
        print("Real trading zatím není zapnutý v této verzi.")

if __name__ == "__main__":
    main()
