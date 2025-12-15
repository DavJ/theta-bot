#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import pandas as pd

from theta_bot_averaging.backtest import run_backtest


def main():
    parser = argparse.ArgumentParser(description="Run backtest on predictions.")
    parser.add_argument("--predictions", required=True, help="CSV or Parquet with predicted_return and signal.")
    parser.add_argument("--fee_rate", type=float, default=0.0004)
    parser.add_argument("--slippage_bps", type=float, default=1.0)
    parser.add_argument("--spread_bps", type=float, default=0.5)
    parser.add_argument("--output", default="runs/manual_backtest", help="Output directory for results.")
    args = parser.parse_args()

    path = Path(args.predictions)
    if path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = pd.read_parquet(path)

    if "predicted_return" not in df.columns:
        raise ValueError("Missing predicted_return. Run inference first or fix pipeline.")
    if "future_return" not in df.columns:
        raise ValueError("Missing future_return column required for backtesting.")
    if "signal" not in df.columns:
        raise ValueError("Missing signal column required for backtesting.")

    output_dir = Path(args.output)
    res = run_backtest(
        df,
        position=df["signal"],
        future_return_col="future_return",
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        spread_bps=args.spread_bps,
        output_dir=output_dir,
    )
    print(json.dumps(res.metrics, indent=2))


if __name__ == "__main__":
    main()
