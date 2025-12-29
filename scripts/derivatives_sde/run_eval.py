#!/usr/bin/env python3
"""Run derivatives SDE evaluation using processed outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from theta_bot_averaging.derivatives_sde_min import compute, evaluate_symbol, loaders, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate derivatives SDE outputs.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., BTCUSDT,ETHUSDT")
    parser.add_argument("--start", help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end", help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--q", type=float, default=0.85)
    parser.add_argument("--horizons", default="1,3,6,12,24", help="Comma-separated horizons in hours")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed/derivatives_sde")
    parser.add_argument("--report", default="reports/DERIVATIVES_SDE_EVAL.md")
    return parser.parse_args()


def load_processed(symbol: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / f"{symbol}_1h.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, compression="gzip", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    processed_dir = Path(args.processed_dir)
    eval_results: Dict[str, Dict] = {}

    for sym in symbols:
        try:
            df = load_processed(sym, processed_dir)
        except FileNotFoundError:
            panel = loaders.load_symbol_panel(sym, interval=args.interval, data_dir=args.data_dir, start=args.start, end=args.end)
            df = compute.compute_mu_sigma_lambda(panel, q=args.q)
            loaders.save_processed(sym, df, processed_dir)

        if args.start or args.end:
            df = df.loc[args.start : args.end]

        eval_results[sym] = evaluate_symbol(df, horizons=horizons, q=args.q)
        print(f"{sym}: evaluated horizons {horizons}")

    report.write_eval_report(eval_results, path=args.report)


if __name__ == "__main__":
    main()
