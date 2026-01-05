from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

if __package__ is None and __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spot_bot.backtest.backtest_spot import run_strategy_backtests
from spot_bot.features import FeatureConfig, compute_features
from spot_bot.persist import SQLiteLogger
from spot_bot.strategies.kalman import KalmanStrategy
from spot_bot.strategies.mean_reversion import MeanReversionStrategy

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
    parser = argparse.ArgumentParser(description="Run spot backtest (mean reversion or Kalman).")
    parser.add_argument("--csv", type=str, default=None, help="Path to OHLCV CSV with a 'close' column.")
    parser.add_argument("--output", type=str, default="backtest_equity.png", help="Optional output plot path.")
    parser.add_argument("--bars", type=int, default=240, help="Synthetic bar count if CSV not provided.")
    parser.add_argument("--slippage-bps", type=float, default=0.5, help="Simple slippage penalty in basis points.")
    parser.add_argument("--strategy", type=str, choices=["meanrev", "kalman"], default="meanrev")
    parser.add_argument("--kalman-mode", type=str, choices=["meanrev", "trend"], default="meanrev")
    parser.add_argument("--kalman-q-level", type=float, default=1e-4)
    parser.add_argument("--kalman-q-trend", type=float, default=1e-6)
    parser.add_argument("--kalman-r", type=float, default=1e-3)
    parser.add_argument("--kalman-k", type=float, default=1.5, help="Sigmoid steepness for exposure conversion.")
    parser.add_argument("--kalman-min-bars", type=int, default=10, help="Minimum bars before emitting exposure.")
    parser.add_argument(
        "--save-plots", type=str, default=None, help="Directory prefix to save diagnostic plots (equity, exposure, S/C_int)."
    )
    parser.add_argument("--db", type=str, default=None, help="Optional SQLite database for logging.")
    args = parser.parse_args()

    logger = SQLiteLogger(args.db) if args.db else None
    ohlcv = _load_ohlcv(args.csv, args.bars)
    feat_cfg = FeatureConfig()
    if args.strategy == "kalman":
        strategy = KalmanStrategy(
            q_level=args.kalman_q_level,
            q_trend=args.kalman_q_trend,
            r=args.kalman_r,
            k=args.kalman_k,
            min_bars=args.kalman_min_bars,
        )
    else:
        strategy = MeanReversionStrategy()

    results = run_strategy_backtests(
        ohlcv_df=ohlcv,
        strategy=strategy,
        logger=logger,
        slippage_bps=args.slippage_bps,
        feature_config=feat_cfg,
    )
    summary = pd.DataFrame({name: res.metrics for name, res in results.items()}).T
    print("\nSummary metrics:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

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

        if args.save_plots:
            plots_dir = pathlib.Path(args.save_plots)
            plots_dir.mkdir(parents=True, exist_ok=True)
            exp_fig = plt.figure(figsize=(10, 3))
            for name, res in results.items():
                if res.exposure is not None and not res.exposure.empty:
                    plt.plot(res.exposure.index, res.exposure.values, label=f"{name} exposure")
            plt.legend()
            plt.title("Exposure over time")
            plt.tight_layout()
            exp_fig.savefig(plots_dir / "exposure.png")
            plt.close(exp_fig)

            risk_fig = plt.figure(figsize=(10, 2))
            for name, res in results.items():
                if res.risk_state is not None:
                    encoded = res.risk_state.map({"OFF": 0, "REDUCE": 0.5, "ON": 1.0})
                    plt.plot(encoded.index, encoded.values, label=f"{name} risk_state")
            plt.legend()
            plt.title("Risk state timeline")
            plt.tight_layout()
            risk_fig.savefig(plots_dir / "risk_state.png")
            plt.close(risk_fig)

            features = compute_features(ohlcv, feat_cfg)
            if not features.empty:
                fig_s, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                ax[0].plot(features.index, features.get("S", pd.Series(dtype=float)), label="S")
                ax[0].legend()
                ax[1].plot(features.index, features.get("C_int", pd.Series(dtype=float)), label="C_int")
                ax[1].legend()
                plt.tight_layout()
                fig_s.savefig(plots_dir / "features.png")
                plt.close(fig_s)
    except ImportError:
        print("matplotlib not installed; skipping plot.")
    finally:
        if logger:
            logger.log_bars(ohlcv.reset_index().rename(columns={"index": "timestamp"}).to_dict(orient="records"))
            last_close = float(ohlcv["close"].iloc[-1]) if "close" in ohlcv.columns else 0.0
            for name, res in results.items():
                if not res.equity_curve.empty:
                    last_ts = res.equity_curve.index[-1]
                    logger.log_equity(
                        timestamp=last_ts,
                        equity_usdt=res.equity_curve.iloc[-1],
                        btc=res.positions.iloc[-1],
                        usdt=float(res.equity_curve.iloc[-1] - res.positions.iloc[-1] * last_close),
                    )
            logger.close()


if __name__ == "__main__":
    main()
