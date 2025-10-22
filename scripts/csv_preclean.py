import argparse, pandas as pd, numpy as np, hashlib, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--csv-time-col", default="time")
    ap.add_argument("--csv-close-col", default="close")
    ap.add_argument("--out", default=None, help="output cleaned CSV (default .tmp/cleaned_<hash>.csv)")
    args = ap.parse_args()

    p = args.csv_path
    if not os.path.exists(p):
        sys.exit(f"[error] CSV not found: {p}")

    df = pd.read_csv(p)
    if args.csv_time_col not in df.columns or args.csv_close_col not in df.columns:
        sys.exit(f"[error] CSV must have columns '{args.csv_time_col}' and '{args.csv_close_col}'")

    df = df[[args.csv_time_col, args.csv_close_col]].rename(columns={args.csv_time_col:"time", args.csv_close_col:"close"})
    df["time"]  = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().sort_values("time").reset_index(drop=True)

    if df["time"].duplicated().any():
        print("[warn] duplicate timestamps detected; keeping first occurrence")
        df = df.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

    if args.out is None:
        os.makedirs(".tmp", exist_ok=True)
        h = hashlib.md5((p + str(len(df))).encode()).hexdigest()[:10]
        args.out = f".tmp/cleaned_{h}.csv"

    df.to_csv(args.out, index=False)
    print(args.out)

if __name__ == "__main__":
    main()
