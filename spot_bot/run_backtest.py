from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from spot_bot.backtest.backtest_spot import run_mean_reversion_backtests

PRICE_NOISE_STD = 0.0005


def _load_ohlcv(csv_path: Optional[str], bars: int) -> pd.DataFrame:
    if csv_path:
        path = pathlib.Path(csv_path)
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        if "close" not in df.columns and "Close" in df.columns:
            df = df.rename(columns={"Close": "close"})
        if "close" not in df.columns:
            raise ValueError("CSV must contain a 'close' column.")
        return df

    np.random.seed(42)
    idx = pd.date_range("2024-01-01", periods=bars, freq="H")
    base = 20000 + np.linspace(0, 500, bars)
    noise = np.sin(np.linspace(0, 6.28, bars)) * 50
    close = base + noise
    open_price_noise = np.random.normal(0, PRICE_NOISE_STD, size=bars)
    spread_noise = np.abs(np.random.normal(0, PRICE_NOISE_STD, size=bars))
    open_ = close * (1 + open_price_noise)
    base_high = np.maximum(open_, close)
    base_low = np.minimum(open_, close)
    high = base_high * (1 + spread_noise)
    low = np.maximum(0.0, base_low * (1 - spread_noise))
    high = np.maximum(np.maximum(high, open_), close)
    low = np.minimum(np.minimum(low, open_), close)
    volume = np.full(bars, 1.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mean reversion spot backtest.")
    parser.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV with a 'close' column.")
    parser.add_argument("--output", type=str, default="backtest_equity.png", help="Optional output plot path.")
    parser.add_argument("--bars", type=int, default=240, help="Synthetic bar count if CSV not provided.")
    args = parser.parse_args()

    ohlcv = _load_ohlcv(args.csv, args.bars)
    results = run_mean_reversion_backtests(ohlcv)

    for name, res in results.items():
        print(f"\n{name.upper()} RESULTS")
        for k, v in res.metrics.items():
            print(f"{k}: {v:.4f}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        for name, res in results.items():
            if not res.equity_curve.empty:
                plt.plot(res.equity_curve.index, res.equity_curve.values, label=name)
        plt.legend()
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Equity (USDT)")
        plt.tight_layout()
        if args.output:
            plt.savefig(args.output)
            print(f"Saved equity plot to {args.output}")
        plt.close()
    except ImportError:
        print("matplotlib not installed; skipping plot.")


if __name__ == "__main__":
    main()
