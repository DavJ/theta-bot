#!/usr/bin/env python3
"""
Benchmark multiple strategies (mean reversion & Kalman) across symbols.

Outputs:
- bench_out/benchmark_strategies.csv (summary per symbol/variant)
- bench_out/benchmark_strategies_pivot.csv (variants as columns)
- bench_out/windows.csv (optional rolling-window metrics)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

if __package__ is None and __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spot_bot.backtest.backtest_spot import _max_drawdown, run_strategy_backtests
from spot_bot.features import FeatureConfig
from spot_bot.strategies.kalman import KalmanStrategy
from spot_bot.strategies.mean_reversion import MeanReversionStrategy

DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "AVAX/USDT",
]


def _load_cached(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_http(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    from download_market_data import download_binance_data

    df = download_binance_data(symbol=symbol.replace("/", ""), interval=timeframe, limit=limit)
    df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_ccxt(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    from theta_features.binance_data import fetch_ohlcv_ccxt

    df = fetch_ohlcv_ccxt(symbol=symbol, timeframe=timeframe, limit_total=limit)
    df = df.rename(columns={"dt": "timestamp"})
    df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    return df[["open", "high", "low", "close", "volume"]]


def load_or_fetch_ohlcv(
    symbol: str, timeframe: str, limit_total: int, workdir: Path, use_cache_only: bool
) -> Tuple[pd.DataFrame, Path]:
    safe = symbol.replace("/", "_")
    cache_path = workdir / f"ohlcv_{safe}.csv"
    if cache_path.exists():
        return _load_cached(cache_path), cache_path
    if use_cache_only:
        raise FileNotFoundError(f"Cached OHLCV not found for {symbol} at {cache_path}")
    try:
        df = _fetch_http(symbol, timeframe, limit_total)
    except Exception:
        df = _fetch_ccxt(symbol, timeframe, limit_total)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(cache_path, index=False)
    return df, cache_path


def _strategy_results(
    ohlcv: pd.DataFrame,
    feature_cfg: FeatureConfig,
    regime_cfg: Dict,
    fee_rate: float,
    max_exposure: float,
    initial_equity: float,
    slippage_bps: float,
    kalman_params: Dict[str, float],
) -> Dict[str, Dict]:
    results = {}

    mr_results = run_strategy_backtests(
        ohlcv_df=ohlcv,
        strategy=MeanReversionStrategy(),
        feature_config=feature_cfg,
        regime_config=regime_cfg,
        fee_rate=fee_rate,
        max_exposure=max_exposure,
        initial_equity=initial_equity,
        slippage_bps=slippage_bps,
    )
    results["meanrev_baseline"] = mr_results["baseline"]
    results["meanrev_gated"] = mr_results["gated"]

    kalman_strategy = KalmanStrategy(**kalman_params)
    kalman_results = run_strategy_backtests(
        ohlcv_df=ohlcv,
        strategy=kalman_strategy,
        feature_config=feature_cfg,
        regime_config=regime_cfg,
        fee_rate=fee_rate,
        max_exposure=max_exposure,
        initial_equity=initial_equity,
        slippage_bps=slippage_bps,
    )
    results["kalman_baseline"] = kalman_results["baseline"]
    results["kalman_gated"] = kalman_results["gated"]
    return results


def _collect_window_rows(
    symbol: str,
    variant: str,
    result,
    window_bars: int,
    window_days: float,
) -> List[Dict]:
    rows: List[Dict] = []
    equity = result.equity_curve
    exposure = result.exposure if result.exposure is not None else pd.Series(dtype=float)
    if equity.empty:
        return rows

    def compute_segment_metrics(eq_seg: pd.Series, exp_seg: pd.Series) -> Dict:
        if eq_seg.empty:
            return {}
        final_return = float(eq_seg.iloc[-1] / max(eq_seg.iloc[0], 1e-8) - 1.0)
        dd = _max_drawdown(eq_seg)
        tim = float(exp_seg.mean()) if not exp_seg.empty else 0.0
        return {"final_return": final_return, "max_drawdown": dd, "time_in_market": tim}

    if window_bars > 0:
        for start in range(0, len(equity) - window_bars + 1, window_bars):
            eq_seg = equity.iloc[start : start + window_bars]
            exp_seg = exposure.iloc[start : start + window_bars] if not exposure.empty else pd.Series(dtype=float)
            metrics = compute_segment_metrics(eq_seg, exp_seg)
            if metrics:
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": variant,
                        "window_start": eq_seg.index[0],
                        "window_end": eq_seg.index[-1],
                        **metrics,
                    }
                )

    if window_days > 0 and isinstance(equity.index, pd.DatetimeIndex):
        start_ts = equity.index[0]
        delta = pd.Timedelta(days=window_days)
        while start_ts < equity.index[-1]:
            end_ts = start_ts + delta
            mask = (equity.index >= start_ts) & (equity.index < end_ts)
            eq_seg = equity.loc[mask]
            exp_seg = exposure.loc[mask] if not exposure.empty else pd.Series(dtype=float)
            metrics = compute_segment_metrics(eq_seg, exp_seg)
            if metrics:
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": variant,
                        "window_start": start_ts,
                        "window_end": end_ts,
                        **metrics,
                    }
                )
            start_ts = end_ts

    return rows


def _stability_score(window_rows: Iterable[Dict]) -> float:
    rows = list(window_rows)
    if not rows:
        return float("nan")
    returns = [float(r["final_return"]) for r in rows]
    drawdowns = [float(r["max_drawdown"]) for r in rows]
    median_return = float(np.median(returns))
    p10_return = float(np.percentile(returns, 10))
    median_dd = float(np.median(drawdowns))
    return median_return - abs(p10_return) - abs(median_dd)


def _flatten_pivot_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(pivot.columns, pd.MultiIndex):
        return pivot
    pivot = pivot.copy()
    pivot.columns = [f"{col[1]}__{col[0]}" for col in pivot.columns]
    return pivot.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark mean reversion vs Kalman strategies across pairs.")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit-total", type=int, default=8000)
    ap.add_argument("--workdir", default="bench_out")
    ap.add_argument("--out", default="bench_out/benchmark_strategies.csv")
    ap.add_argument("--pivot-out", default="bench_out/benchmark_strategies_pivot.csv")
    ap.add_argument("--fee-rate", type=float, default=0.0005)
    ap.add_argument("--slippage-bps", type=float, default=0.5)
    ap.add_argument("--max-exposure", type=float, default=1.0)
    ap.add_argument("--initial-equity", type=float, default=1000.0)
    ap.add_argument("--window-days", type=float, default=0.0, help="Rolling window in days (0 to disable).")
    ap.add_argument("--window-bars", type=int, default=0, help="Rolling window in bars (0 to disable).")
    ap.add_argument("--use-cache-only", action="store_true", help="Fail if cached OHLCV is missing instead of downloading.")
    ap.add_argument("--kalman-q-level", type=float, default=1e-4)
    ap.add_argument("--kalman-q-trend", type=float, default=1e-6)
    ap.add_argument("--kalman-r", type=float, default=1e-3)
    ap.add_argument("--kalman-k", type=float, default=1.5)
    ap.add_argument("--kalman-min-bars", type=int, default=10)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    feature_cfg = FeatureConfig()
    regime_cfg: Dict = {}
    kalman_params = {
        "q_level": args.kalman_q_level,
        "q_trend": args.kalman_q_trend,
        "r": args.kalman_r,
        "k": args.kalman_k,
        "min_bars": args.kalman_min_bars,
    }

    summary_rows: List[Dict] = []
    window_rows: List[Dict] = []

    for symbol in symbols:
        ohlcv, cache_path = load_or_fetch_ohlcv(
            symbol=symbol,
            timeframe=args.timeframe,
            limit_total=args.limit_total,
            workdir=workdir,
            use_cache_only=args.use_cache_only,
        )
        if ohlcv.empty:
            continue

        results = _strategy_results(
            ohlcv=ohlcv,
            feature_cfg=feature_cfg,
            regime_cfg=regime_cfg,
            fee_rate=args.fee_rate,
            max_exposure=args.max_exposure,
            initial_equity=args.initial_equity,
            slippage_bps=args.slippage_bps,
            kalman_params=kalman_params,
        )

        for variant, res in results.items():
            row = {"symbol": symbol, "variant": variant, **res.metrics}
            summary_rows.append(row)
            if args.window_bars or args.window_days:
                window_rows.extend(
                    _collect_window_rows(
                        symbol=symbol,
                        variant=variant,
                        result=res,
                        window_bars=max(0, int(args.window_bars)),
                        window_days=max(0.0, float(args.window_days)),
                    )
                )

    summary = pd.DataFrame(summary_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Saved summary CSV to {out_path}")

    # Pivot for quick comparison (flattened columns variant__metric)
    pivot_values = [c for c in ("final_return", "max_drawdown", "time_in_market") if c in summary.columns]
    if pivot_values and not summary.empty:
        pivot = summary.pivot_table(index="symbol", columns="variant", values=pivot_values)
        pivot_flat = _flatten_pivot_columns(pivot)
        pivot_path = Path(args.pivot_out)
        pivot_path.parent.mkdir(parents=True, exist_ok=True)
        pivot_flat.to_csv(pivot_path, index=False)
        print(f"Saved pivot CSV to {pivot_path}")
    else:
        print("No metrics available to pivot; skipping pivot export.")

    if window_rows:
        windows_df = pd.DataFrame(window_rows)
        windows_path = workdir / "windows.csv"
        windows_df.to_csv(windows_path, index=False)
        print(f"Saved window analysis to {windows_path}")

        # Stability score
        stability = []
        for variant in summary["variant"].unique():
            subset = [r for r in window_rows if r["variant"] == variant]
            stability.append({"variant": variant, "stability_score": _stability_score(subset)})
        stability_df = pd.DataFrame(stability)
        summary = summary.merge(stability_df, on="variant", how="left")
        summary.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
