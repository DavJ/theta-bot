#!/usr/bin/env python3
import argparse, pandas as pd, os, hashlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--csv-time-col', default='time')
    ap.add_argument('--csv-close-col', default='close')
    ap.add_argument('--outdir', default='.tmp')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.csv_time_col not in df.columns or args.csv_close_col not in df.columns:
        raise SystemExit(f"[err] CSV must have columns {args.csv_time_col},{args.csv_close_col}. Found {list(df.columns)}")
    df = df.rename(columns={args.csv_time_col:'time', args.csv_close_col:'close'})[['time','close']]
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna().sort_values('time').reset_index(drop=True)

    os.makedirs(args.outdir, exist_ok=True)
    h = hashlib.md5((args.csv + str(len(df))).encode()).hexdigest()[:10]
    out = os.path.join(args.outdir, f'cleaned_{h}.csv')
    df.to_csv(out, index=False)
    print(f"[ok] cleaned CSV -> {out}")

if __name__ == "__main__":
    main()
