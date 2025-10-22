import argparse, subprocess, sys, os, shlex

def main():
    ap = argparse.ArgumentParser(description="CSV-safe runner for theta_eval_hbatch_biquat.py")
    ap.add_argument("--symbols", required=True, help="CSV path nebo seznam symbolů")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--minP", type=int, default=24)
    ap.add_argument("--maxP", type=int, default=480)
    ap.add_argument("--nP", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--lambda", dest="lam", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--phase", choices=["simple","complex","biquat"], default="biquat")
    ap.add_argument("--out", required=True)
    # CSV overrides
    ap.add_argument("--csv-time-col", default="time")
    ap.add_argument("--csv-close-col", default="close")
    # Cesta k původnímu evaluátoru
    ap.add_argument("--evaluator", default="theta_eval_hbatch_biquat.py")
    args = ap.parse_args()

    symbols = args.symbols
    cleaned = symbols
    if symbols.lower().endswith(".csv"):
        cmd_clean = [
            sys.executable, "scripts/csv_preclean.py", symbols,
            "--csv-time-col", args.csv_time_col,
            "--csv-close-col", args.csv_close_col
        ]
        print("+", " ".join(shlex.quote(x) for x in cmd_clean))
        cleaned = subprocess.check_output(cmd_clean, text=True).strip()
        print(f"[ok] cleaned CSV -> {cleaned}")

    cmd = [
        sys.executable, args.evaluator,
        "--symbols", cleaned,
        "--interval", args.interval,
        "--window", str(args.window),
        "--horizon", str(args.horizon),
        "--minP", str(args.minP),
        "--maxP", str(args.maxP),
        "--nP", str(args.nP),
        "--sigma", str(args.sigma),
        "--lambda", str(args.lam),
        "--limit", str(args.limit),
        "--phase", args.phase,
        "--out", args.out,
    ]

    print("+", " ".join(shlex.quote(x) for x in cmd))
    rc = subprocess.call(cmd)
    sys.exit(rc)

if __name__ == "__main__":
    main()
