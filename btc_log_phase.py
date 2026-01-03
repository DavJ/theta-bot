"""
BTC/USDT log-phase (fractional log) regime feature and quick backtest.

The idea: treat the fractional part of log returns as a circular phase.
On the unit circle, clustered phases imply structure/regime; uniform phases
look like noise. We measure clustering via the mean resultant length of
complex unit vectors (exp(i*2*pi*phi)) over a rolling window.
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest


def fetch_ohlcv_binance(
    symbol: str = "BTC/USDT", timeframe: str = "1h", limit_total: int = 6000
) -> pd.DataFrame:
    """
    Download OHLCV from Binance with pagination and return as a DataFrame.

    Columns: ts, open, high, low, close, volume, dt (UTC datetime).
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - (limit_total + 10) * tf_ms
    data = []

    while len(data) < limit_total:
        limit = min(1000, limit_total - len(data))
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not batch:
            break
        data.extend(batch)
        since = batch[-1][0] + tf_ms

    df = pd.DataFrame(
        data, columns=["ts", "open", "high", "low", "close", "volume"]
    ).drop_duplicates(subset="ts")
    df = df.sort_values("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def frac(x: np.ndarray | float) -> np.ndarray:
    """Fractional part in [0,1) for real inputs."""
    arr = np.asarray(x, dtype=float)
    return arr - np.floor(arr)


def log_phase(x: np.ndarray | float, base: float = 10.0, eps: float = 1e-12) -> np.ndarray:
    """Phase on [0,1) using fractional part of log-base returns."""
    arr = np.asarray(x, dtype=float)
    arr = np.maximum(arr, eps)
    return frac(np.log(arr) / math.log(base))


def circ_dist(a: float, b: float) -> float:
    """Circular distance on S1."""
    d = abs(a - b)
    return min(d, 1.0 - d)


def phase_embedding(phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map phase to unit circle coordinates (cos, sin)."""
    phi_arr = np.asarray(phi, dtype=float)
    angles = 2 * np.pi * phi_arr
    return np.cos(angles), np.sin(angles)


def rolling_phase_concentration(phi: np.ndarray, window: int = 256) -> np.ndarray:
    """Rolling mean resultant length |E[e^{i*2*pi*phi}]|."""
    phi_series = pd.Series(phi, dtype=float)
    angles = 2 * np.pi * phi_series
    cos_part = np.cos(angles)
    sin_part = np.sin(angles)
    mean_cos = cos_part.rolling(window=window, min_periods=window).mean()
    mean_sin = sin_part.rolling(window=window, min_periods=window).mean()
    return np.sqrt(mean_cos**2 + mean_sin**2).to_numpy()


def uniformity_test(phi: np.ndarray) -> Tuple[float, float]:
    """KS test vs uniform(0,1) as a quick diagnostic (phi is circular)."""
    phi_arr = np.asarray(phi, dtype=float)
    phi_arr = phi_arr[~np.isnan(phi_arr)]
    if phi_arr.size == 0:
        return math.nan, math.nan
    stat, pvalue = kstest(phi_arr, "uniform")
    return float(stat), float(pvalue)


def max_drawdown(equity: np.ndarray) -> float:
    equity_arr = np.asarray(equity, dtype=float)
    if equity_arr.size == 0:
        return math.nan
    running_max = np.maximum.accumulate(equity_arr)
    dd = equity_arr / running_max - 1.0
    return float(dd.min())


def risk_filter_backtest(df_feat: pd.DataFrame, thr: float = 0.20):
    """
    Simple risk/regime filter.
    Weights are flat (0) when concentration exceeds thr, else fully invested.
    """
    rets = df_feat["ret"].to_numpy()
    rets = np.nan_to_num(rets, nan=1.0)
    weights = np.where(df_feat["concentration"] <= thr, 1.0, 0.0)

    eq_bh = np.cumprod(rets)
    eq_filtered = np.cumprod(1.0 + weights * (rets - 1.0))

    summary = {
        "final_return_bh": float(eq_bh[-1] - 1.0) if eq_bh.size else math.nan,
        "final_return_filtered": float(eq_filtered[-1] - 1.0)
        if eq_filtered.size
        else math.nan,
        "max_drawdown_bh": max_drawdown(eq_bh),
        "max_drawdown_filtered": max_drawdown(eq_filtered),
        "time_in_market": float(weights.mean()) if weights.size else math.nan,
    }
    return eq_bh, eq_filtered, summary


def _plot_outputs(df: pd.DataFrame, save_plots: bool = False) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    axes[0].hist(df["phi"].dropna(), bins=60, color="steelblue", alpha=0.8)
    axes[0].set_title("Histogram of log-phase (phi)")
    axes[0].set_ylabel("Count")

    axes[1].plot(df["dt"], df["concentration"], color="darkorange", linewidth=1.2)
    axes[1].set_title("Rolling phase concentration |E[e^(i*2*pi*phi)]|")
    axes[1].set_ylabel("Concentration")

    axes[2].plot(df["dt"], df["eq_bh"], label="Buy & Hold", linewidth=1.2)
    axes[2].plot(df["dt"], df["eq_filtered"], label="Filtered", linewidth=1.2)
    axes[2].set_title("Equity curves")
    axes[2].set_ylabel("Equity")
    axes[2].legend()

    plt.tight_layout()
    if save_plots:
        plt.savefig("btc_log_phase.png", dpi=150)
    else:
        plt.show()


def main() -> None:
    symbol = "BTC/USDT"
    timeframe = "1h"
    limit_total = 6000
    base = 10.0
    window = 256
    thr = 0.20

    parser = argparse.ArgumentParser(description="BTC log-phase regime filter demo")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to PNG")
    args = parser.parse_args()

    df = fetch_ohlcv_binance(symbol=symbol, timeframe=timeframe, limit_total=limit_total)
    df["ret"] = df["close"] / df["close"].shift(1)
    df["phi"] = log_phase(df["ret"], base=base)
    df["cos_phi"], df["sin_phi"] = phase_embedding(df["phi"])
    df["concentration"] = rolling_phase_concentration(df["phi"], window=window)

    ks_stat, ks_p = uniformity_test(df["phi"])
    print(f"KS uniformity test: stat={ks_stat:.4f}, p-value={ks_p:.4f}")

    conc_clean = df["concentration"].dropna()
    if not conc_clean.empty:
        print(
            f"Concentration median={conc_clean.median():.4f}, "
            f"95th percentile={conc_clean.quantile(0.95):.4f}"
        )

    eq_bh, eq_filtered, summary = risk_filter_backtest(df, thr=thr)
    df["eq_bh"] = eq_bh
    df["eq_filtered"] = eq_filtered

    print(
        "Backtest summary:",
        f"BH final={summary['final_return_bh']:.2%},",
        f"Filtered final={summary['final_return_filtered']:.2%},",
        f"BH maxDD={summary['max_drawdown_bh']:.2%},",
        f"Filtered maxDD={summary['max_drawdown_filtered']:.2%},",
        f"Time in market={summary['time_in_market']:.1%}",
    )

    _plot_outputs(df, save_plots=args.save_plots)


if __name__ == "__main__":
    main()
