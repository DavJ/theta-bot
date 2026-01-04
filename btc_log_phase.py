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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest

from theta_features.binance_data import fetch_ohlcv_ccxt
from theta_features.log_phase_core import (
    circ_dist,
    frac,
    log_phase,
    max_drawdown,
    phase_embedding,
    rolling_phase_concentration,
)


def fetch_ohlcv_binance(
    symbol: str = "BTC/USDT", timeframe: str = "1h", limit_total: int = 6000
) -> pd.DataFrame:
    """
    Download OHLCV from Binance with pagination and return as a DataFrame.

    Columns: ts, open, high, low, close, volume, dt (UTC datetime).
    """
    return fetch_ohlcv_ccxt(symbol=symbol, timeframe=timeframe, limit_total=limit_total)


def uniformity_test(phi: np.ndarray) -> Tuple[float, float]:
    """KS test vs uniform(0,1) as a quick diagnostic (phi is circular)."""
    phi_arr = np.asarray(phi, dtype=float)
    phi_arr = phi_arr[~np.isnan(phi_arr)]
    if phi_arr.size == 0:
        return math.nan, math.nan
    stat, pvalue = kstest(phi_arr, "uniform")
    return float(stat), float(pvalue)


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
