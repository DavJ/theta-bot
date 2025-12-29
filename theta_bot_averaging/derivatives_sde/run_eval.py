#!/usr/bin/env python3
"""CLI for evaluating deterministic bias."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .eval import evaluate_bias
from .report import evaluation_report, write_report
from .sde_decompose import gate_lambda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate derivatives SDE bias.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols to evaluate")
    parser.add_argument("--horizons", default="1,3,6,12,24", help="Comma-separated horizons in hours")
    parser.add_argument("--tau_quantile", type=float, default=None, help="Optional quantile gating override")
    parser.add_argument("--processed_dir", default="data/processed/derivatives_sde")
    parser.add_argument("--out", default="reports/DERIVATIVES_SDE_EVAL.md")
    return parser.parse_args()


def _load_processed(symbol: str, processed_dir: str) -> pd.DataFrame:
    path = Path(processed_dir) / f"{symbol}_1h.csv.gz"
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    return df


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    horizons = [int(h) for h in args.horizons.split(",") if h]

    eval_results = {}
    for sym in symbols:
        df = _load_processed(sym, args.processed_dir)
        if args.tau_quantile is not None and "Lambda" in df.columns:
            df["active"] = gate_lambda(df["Lambda"], tau=None, tau_quantile=args.tau_quantile)
        eval_results[sym] = evaluate_bias(df, horizons=horizons)

    content = evaluation_report(eval_results)
    write_report(content, args.out)
    print(f"Evaluation report written to {args.out}")


if __name__ == "__main__":
    main()
