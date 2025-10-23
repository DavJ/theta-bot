#!/usr/bin/env python3
import argparse, subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "theta_bot_averaging" / "theta_eval_hbatch_biquat_max.py"

def ensure_csv_for_symbol(symbol, interval, limit=1000):
    """If symbol is a ticker, download to prices/<TICKER>_<interval>.csv"""
    if symbol.lower().endswith(".csv"):
        return symbol
    maker = ROOT / "make_prices_csv.py"
    if not maker.exists():
        raise SystemExit("make_prices_csv.py nebyl nalezen, nelze stáhnout ceny.")
    out_csv = ROOT / "prices" / f"{symbol.upper()}_{interval}.csv"
    out_csv.parent.mkdir(exist_ok=True, parents=True)
    cmd = [sys.executable, str(maker),
           "--symbols", symbol.upper(),
           "--interval", interval,
           "--limit", str(limit),
           "--outdir", "prices"]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return str(out_csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="Ticker (BTCUSDT) nebo CSV cesta (prices/...csv)")
    ap.add_argument("--interval", required=True)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--minP", type=int, default=24)
    ap.add_argument("--maxP", type=int, default=480)
    ap.add_argument("--nP", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--lam", dest="lam", type=float, default=1e-3)
    ap.add_argument("--pred-ensemble", default="avg", choices=["avg", "max"])
    ap.add_argument("--max-by", default="transform", choices=["transform", "contrib"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    csv_path = ensure_csv_for_symbol(args.symbol, args.interval, args.limit)
    print(f"\n=== Running {csv_path} ===\n")

    if not EVAL_SCRIPT.exists():
        raise SystemExit(f"Evaluator {EVAL_SCRIPT} nebyl nalezen.")

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--symbols", csv_path,
        "--interval", args.interval,
        "--window", str(args.window),
        "--horizon", str(args.horizon),
        "--minP", str(args.minP),
        "--maxP", str(args.maxP),
        "--nP", str(args.nP),
        "--sigma", str(args.sigma),
        "--lambda", str(args.lam),
        "--pred-ensemble", args.pred_ensemble,
        "--max-by", args.max_by,
        "--out", "results/hbatch_biquat_summary_live.csv",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    if args.dry_run:
        print("[dry-run] Hotovo (predikce v eval_h_*.csv).")
        return

    print("TODO: sem přijde real-time exekuce (napojená na burzu).")

if __name__ == "__main__":
    main()
