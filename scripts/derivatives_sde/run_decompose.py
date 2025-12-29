#!/usr/bin/env python3
"""Run derivatives SDE decomposition (minimal pipeline)."""

from __future__ import annotations

import argparse
from typing import List

from theta_bot_averaging.derivatives_sde_min import compute, loaders, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run derivatives SDE decomposition.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., BTCUSDT,ETHUSDT")
    parser.add_argument("--start", help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end", help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--z_window", type=int, default=168)
    parser.add_argument("--sigma_window", type=int, default=168)
    parser.add_argument("--q", type=float, default=0.85, help="Quantile for Lambda gating")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed/derivatives_sde")
    parser.add_argument("--report", default="reports/DERIVATIVES_SDE_DECOMPOSITION.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    summaries: List[dict] = []

    for sym in symbols:
        panel = loaders.load_symbol_panel(sym, interval=args.interval, data_dir=args.data_dir, start=args.start, end=args.end)
        df = compute.compute_mu_sigma_lambda(
            panel,
            alpha=args.alpha,
            beta=args.beta,
            z_window=args.z_window,
            sigma_window=args.sigma_window,
            q=args.q,
        )
        out_path = loaders.save_processed(sym, df, args.out_dir)

        if df.empty:
            summaries.append(
                {
                    "symbol": sym,
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "active_share": 0.0,
                    "lambda_threshold": float("nan"),
                    "top_events": [],
                }
            )
            continue

        top = df.nlargest(5, "lambda")[["lambda", "mu", "sigma"]]
        top_events = [
            {
                "timestamp": int(idx.timestamp() * 1000),
                "lambda": row["lambda"],
                "mu": row["mu"],
                "sigma": row["sigma"],
            }
            for idx, row in top.iterrows()
        ]
        lambda_non_null = df["lambda_threshold"].dropna()
        lambda_threshold_value = float(lambda_non_null.iloc[-1]) if len(lambda_non_null) else float("nan")
        summaries.append(
            {
                "symbol": sym,
                "rows": len(df),
                "start": df.index[0],
                "end": df.index[-1],
                "active_share": float(df["active"].mean()) if len(df) else 0.0,
                "lambda_threshold": lambda_threshold_value,
                "top_events": top_events,
            }
        )
        print(f"{sym}: wrote decomposition with {len(df)} rows to {out_path}")

    report.write_decomposition_report(summaries, path=args.report)


if __name__ == "__main__":
    main()
