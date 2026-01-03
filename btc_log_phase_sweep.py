"""
Multi-candidate log-phase sweep for BTC/USDT (or any symbol).

Build multiple positive input series x_t from OHLCV, compute log-phase
features, and evaluate their predictive value against future targets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from btc_log_phase import (
    fetch_ohlcv_binance,
    log_phase,
    phase_embedding,
    risk_filter_backtest,
    rolling_phase_concentration,
    uniformity_test,
)

DEFAULT_CANDIDATES = [
    "price",
    "abslogret",
    "rv",
    "atr",
    "volume",
    "distema",
]


def build_candidate_series(df: pd.DataFrame, name: str, args: argparse.Namespace) -> pd.Series:
    """
    Construct candidate positive series x_t from OHLCV data.
    The output aligns with df index and may contain NaNs from rolling windows.
    """
    name = name.lower()
    rv_window = getattr(args, "rv_window", 24)
    atr_window = getattr(args, "atr_window", 14)
    ema_window = getattr(args, "ema_window", 50)
    if name == "price":
        series = df["close"]
    elif name == "abslogret":
        lr = np.log(df["close"] / df["close"].shift(1))
        series = lr.abs()
    elif name == "rv":
        lr = np.log(df["close"] / df["close"].shift(1))
        series = lr.rolling(window=rv_window, min_periods=rv_window).std()
    elif name == "atr":
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        series = tr.rolling(window=atr_window, min_periods=atr_window).mean()
    elif name == "volume":
        series = df["volume"]
        if getattr(args, "volume_roll", 0):
            series = series.rolling(
                window=args.volume_roll, min_periods=args.volume_roll
            ).sum()
    elif name == "distema":
        ema = df["close"].ewm(span=ema_window, adjust=False).mean()
        series = (df["close"] - ema).abs()
    else:
        raise ValueError(f"Unknown candidate: {name}")

    series = series.replace([np.inf, -np.inf], np.nan)
    return series


def build_targets(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Construct strictly future-aligned targets."""
    target_window = getattr(args, "target_window", 24)
    horizon = getattr(args, "horizon", 24)
    lr = np.log(df["close"] / df["close"].shift(1))
    future_lr = lr.shift(-1)
    future_vol = (
        future_lr.rolling(window=target_window, min_periods=target_window)
        .std()
        .shift(-(target_window - 1))
    )
    future_absret = (np.log(df["close"].shift(-horizon) / df["close"])).abs()
    return pd.DataFrame({"y_vol": future_vol, "y_absret": future_absret})


def compute_features(x: pd.Series, args: argparse.Namespace) -> pd.DataFrame:
    """Compute log-phase derived features for a candidate series."""
    base = getattr(args, "base", 10.0)
    conc_window = getattr(args, "conc_window", 256)
    phi = log_phase(x.to_numpy(), base=base)
    cos_phi, sin_phi = phase_embedding(phi)
    concentration = rolling_phase_concentration(phi, window=conc_window)
    return pd.DataFrame(
        {
            "phi": phi,
            "cos_phi": cos_phi,
            "sin_phi": sin_phi,
            "concentration": concentration,
        },
        index=x.index,
    )


def _bucket_stats(feature: pd.Series, target: pd.Series, buckets: int = 5) -> Dict[str, object]:
    """Return bucket means, counts, ratio, and monotonicity."""
    out: Dict[str, object] = {"bucket_counts": {}, "bucket_means": {}, "bucket_ratio": math.nan}
    try:
        bucketed = pd.qcut(feature, buckets, labels=False, duplicates="drop")
    except ValueError:
        return out
    means = target.groupby(bucketed).mean()
    counts = bucketed.value_counts().sort_index()
    out["bucket_counts"] = counts.to_dict()
    out["bucket_means"] = means.to_dict()
    if means.empty or means.min() == 0:
        out["bucket_ratio"] = math.nan
    else:
        out["bucket_ratio"] = float(means.max() / means.min())
    diffs = means.diff().dropna()
    if diffs.empty:
        out["bucket_monotonic"] = "flat"
    elif (diffs >= 0).all():
        out["bucket_monotonic"] = "increasing"
    elif (diffs <= 0).all():
        out["bucket_monotonic"] = "decreasing"
    else:
        out["bucket_monotonic"] = "mixed"
    return out


def evaluate_candidate(features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, object]:
    """Compute ICs, uniformity, bucket stats, and optional AUC for a candidate."""
    joined = pd.concat(
        [features[["phi", "cos_phi", "sin_phi", "concentration"]], targets], axis=1
    ).dropna()

    metrics: Dict[str, object] = {
        "ks_stat": math.nan,
        "ks_p": math.nan,
        "conc_median": math.nan,
        "conc_p95": math.nan,
        "ic_conc_y_vol": math.nan,
        "ic_conc_y_absret": math.nan,
        "ic_cos_y_vol": math.nan,
        "ic_sin_y_vol": math.nan,
        "bucket_counts": {},
        "bucket_means": {},
        "bucket_ratio": math.nan,
        "bucket_monotonic": None,
        "auc_conc_y_vol": math.nan,
    }

    if joined.empty:
        return metrics

    metrics["ks_stat"], metrics["ks_p"] = uniformity_test(joined["phi"].to_numpy())
    metrics["conc_median"] = float(joined["concentration"].median())
    metrics["conc_p95"] = float(joined["concentration"].quantile(0.95))
    metrics["ic_conc_y_vol"] = joined["concentration"].corr(joined["y_vol"], method="spearman")
    metrics["ic_conc_y_absret"] = joined["concentration"].corr(
        joined["y_absret"], method="spearman"
    )
    metrics["ic_cos_y_vol"] = joined["cos_phi"].corr(joined["y_vol"], method="spearman")
    metrics["ic_sin_y_vol"] = joined["sin_phi"].corr(joined["y_vol"], method="spearman")

    bucket_out = _bucket_stats(joined["concentration"], joined["y_vol"])
    metrics.update(bucket_out)

    # Optional classification: top 20% vol
    try:
        threshold = joined["y_vol"].quantile(0.8)
        y_class = (joined["y_vol"] >= threshold).astype(int)
        if y_class.nunique() > 1:
            metrics["auc_conc_y_vol"] = roc_auc_score(y_class, joined["concentration"])
    except Exception:
        metrics["auc_conc_y_vol"] = math.nan

    return metrics


def rank_candidates(results: Iterable[Dict[str, object]]) -> pd.DataFrame:
    """Return a summary DataFrame sorted by absolute IC on y_vol."""
    table = pd.DataFrame(results)
    if table.empty:
        return table
    table = table.copy()
    table["abs_ic_conc_y_vol"] = table["ic_conc_y_vol"].abs()
    return table.sort_values(by=["abs_ic_conc_y_vol", "bucket_ratio"], ascending=False)


def _plot_candidate(
    df: pd.DataFrame,
    features: pd.DataFrame,
    candidate: str,
    eq_bh: np.ndarray,
    eq_filtered: np.ndarray,
    save: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    axes[0].hist(features["phi"].dropna(), bins=60, color="steelblue", alpha=0.8)
    axes[0].set_title(f"Histogram of phi - {candidate}")
    axes[0].set_ylabel("Count")

    axes[1].plot(df["dt"], features["concentration"], color="darkorange", linewidth=1.2)
    axes[1].set_title(f"Rolling concentration {candidate}")
    axes[1].set_ylabel("Concentration")

    axes[2].plot(df["dt"], eq_bh, label="Buy & Hold", linewidth=1.2)
    axes[2].plot(df["dt"], eq_filtered, label="Filtered", linewidth=1.2)
    axes[2].set_title(f"Equity curves - {candidate}")
    axes[2].set_ylabel("Equity")
    axes[2].legend()

    plt.tight_layout()
    if save:
        Path("plots").mkdir(exist_ok=True)
        plt.savefig(Path("plots") / f"{candidate}_log_phase.png", dpi=150)
    else:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log-phase sweep over multiple candidates.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit-total", type=int, default=6000)
    parser.add_argument("--base", type=float, default=10.0)
    parser.add_argument("--conc-window", type=int, default=256)
    parser.add_argument("--thr", type=float, default=0.20)
    parser.add_argument("--rv-window", type=int, default=24)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--ema-window", type=int, default=50)
    parser.add_argument("--target-window", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--volume-roll", type=int, default=0, help="Rolling window for volume.")
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated list of candidates. Default runs all.",
    )
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def _candidate_list(args: argparse.Namespace) -> List[str]:
    if args.candidates:
        return [c.strip() for c in args.candidates.split(",") if c.strip()]
    return DEFAULT_CANDIDATES


def main() -> None:
    args = parse_args()
    df = fetch_ohlcv_binance(
        symbol=args.symbol, timeframe=args.timeframe, limit_total=args.limit_total
    )
    targets = build_targets(df, args)
    results = []

    for cand in _candidate_list(args):
        x = build_candidate_series(df, cand, args)
        features = compute_features(x, args)

        metrics = evaluate_candidate(features, targets)
        metrics["candidate"] = cand

        bt_df = pd.DataFrame(
            {
                "ret": df["close"] / df["close"].shift(1),
                "concentration": features["concentration"],
            }
        )
        eq_bh, eq_filtered, bt_summary = risk_filter_backtest(bt_df, thr=args.thr)
        metrics["backtest"] = bt_summary
        results.append(metrics)

        print(
            f"[{cand}] KS={metrics['ks_stat']:.3f}/{metrics['ks_p']:.3f} "
            f"IC(C,y_vol)={metrics['ic_conc_y_vol']:.3f} "
            f"IC(C,y_absret)={metrics['ic_conc_y_absret']:.3f} "
            f"Bucket ratio={metrics['bucket_ratio']:.3f} "
            f"AUC={metrics['auc_conc_y_vol']:.3f}"
        )

        if args.save_plots:
            _plot_candidate(df, features, cand, eq_bh, eq_filtered, save=True)

    ranked = rank_candidates(results)
    if not ranked.empty:
        cols = [
            "candidate",
            "ic_conc_y_vol",
            "ic_conc_y_absret",
            "bucket_ratio",
            "auc_conc_y_vol",
            "conc_median",
            "conc_p95",
        ]
        display_cols = [c for c in cols if c in ranked.columns]
        print("\nRanked summary (by |IC| on y_vol):")
        print(ranked[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


if __name__ == "__main__":
    main()
