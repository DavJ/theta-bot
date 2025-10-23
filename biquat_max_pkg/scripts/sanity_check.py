#!/usr/bin/env python3
import argparse, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--time-col', default='time')
    ap.add_argument('--close-col', default='close')
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=[args.time_col])
    df = df.rename(columns={args.time_col:'time', args.close_col:'close'}).sort_values('time')
    ret = df['close'].pct_change()
    gaps = df['time'].diff().dropna().value_counts().sort_index()
    print("rows:", len(df), "span:", df.time.min(), "→", df.time.max())
    print("dupes:", df.time.duplicated().sum())
    print("\nΔt unique (last 10):")
    print(gaps.tail(10))
    print("\nreturn stats:\n", ret.describe(percentiles=[.01,.05,.95,.99]).to_string())

if __name__ == "__main__":
    main()
