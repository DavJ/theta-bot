#!/usr/bin/env python3
"""CLI for running derivatives SDE decomposition."""

from __future__ import annotations

import argparse

from .sde_decompose import decompose_symbol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run derivatives SDE decomposition.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols, e.g., BTCUSDT,ETHUSDT")
    parser.add_argument("--start", help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end", help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--z_window", type=int, default=168)
    parser.add_argument("--sigma_window", type=int, default=168)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--tau_quantile", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed/derivatives_sde")
    parser.add_argument("--ewma_lambda", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        df = decompose_symbol(
            symbol=sym,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            start=args.start,
            end=args.end,
            z_window=args.z_window,
            sigma_window=args.sigma_window,
            tau=args.tau,
            tau_quantile=args.tau_quantile,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            ewma_lambda=args.ewma_lambda,
        )
        print(f"{sym}: wrote decomposition with {len(df)} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
