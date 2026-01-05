#!/usr/bin/env python3
"""
Benchmark mean reversion and Kalman strategies across symbols/psi_modes.
Produces a single CSV of metrics plus per-run window summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:  # optional plotting dependency
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from spot_bot.features import FeatureConfig, compute_features
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies import KalmanRiskStrategy, MeanRevGatedStrategy, apply_risk_gating, params_hash


DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "AVAX/USDT"]
DEFAULT_PSI_MODES = ["none", "mellin_cepstrum", "mellin_complex_cepstrum"]
ANNUAL_HOURS = 24 * 365


def _load_cached(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_ccxt(symbol: str, timeframe: str, limit_total: int) -> pd.DataFrame:
    try:
        from theta_features.binance_data import fetch_ohlcv_ccxt
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        missing = getattr(exc, "name", "dependency")
        print(f"Missing dependency ({missing}); install ccxt and theta_features to download live data.")
        raise SystemExit(1) from exc
    df = fetch_ohlcv_ccxt(symbol=symbol, timeframe=timeframe, limit_total=limit_total)
    df = df.rename(columns={"dt": "timestamp"})
    df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    return df[["open", "high", "low", "close", "volume"]]


def load_or_fetch(symbol: str, timeframe: str, limit_total: int, workdir: Path, use_cache_only: bool) -> Tuple[pd.DataFrame, Path]:
    safe = symbol.replace("/", "_")
    cache_path = workdir / f"ohlcv_{safe}.csv"
    if cache_path.exists():
        return _load_cached(cache_path), cache_path
    if use_cache_only:
        raise FileNotFoundError(f"Missing cached OHLCV for {symbol} at {cache_path}")
    df = _fetch_ccxt(symbol, timeframe, limit_total)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_csv(cache_path, index=False)
    return df, cache_path


def _calc_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().shift(-1).fillna(0.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min())


def _best_worst_windows(equity: pd.Series, window: int, top_n: int = 3) -> Tuple[List[dict], List[dict]]:
    if equity.empty or window <= 1 or len(equity) < window:
        return [], []
    rows = []
    for start in range(0, len(equity) - window + 1):
        seg = equity.iloc[start : start + window]
        ret = float(seg.iloc[-1] / seg.iloc[0] - 1.0)
        dd = _max_drawdown(seg)
        rows.append({"start": seg.index[0], "end": seg.index[-1], "return": ret, "max_drawdown": dd})
    best = sorted(rows, key=lambda r: r["return"], reverse=True)[:top_n]
    worst = sorted(rows, key=lambda r: r["return"])[:top_n]
    return best, worst


@dataclass
class BacktestResult:
    equity: pd.Series
    exposure: pd.Series
    trades: int
    turnover: float
    fee_paid: float
    metrics: Dict[str, float]
    best_windows: List[dict]
    worst_windows: List[dict]


def run_backtest(
    ohlcv: pd.DataFrame,
    strategy_name: str,
    psi_mode: str,
    feature_cfg: FeatureConfig,
    fee_rate: float,
    slippage_bps: float,
    max_exposure: float,
    initial_equity: float,
    window_bars: int,
    kalman_mode: str = "meanrev",
) -> BacktestResult:
    prices = ohlcv["close"].astype(float)
    returns = _calc_returns(prices)
    strategy_map = {
        "meanrev": MeanRevGatedStrategy(max_exposure=max_exposure),
        "kalman": KalmanRiskStrategy(mode=kalman_mode, max_exposure=max_exposure),
    }
    strategy = strategy_map.get(strategy_name)
    if strategy is None:
        raise ValueError(f"Unsupported strategy: {strategy_name}")
    feat = compute_features(ohlcv, cfg=feature_cfg)
    regime = RegimeEngine({})
    desired = []
    gated = []
    exposures = []
    fee_paid = 0.0
    trades = 0
    turnover = 0.0
    equity = [initial_equity]
    for i in range(len(prices) - 1):
        window_prices = prices.iloc[: i + 1]
        desired_exposure = strategy.generate(window_prices).desired_exposure
        risk_state = "ON"
        risk_budget = 1.0
        if not feat.empty and i < len(feat):
            try:
                dec = regime.decide(feat.iloc[: i + 1])
                risk_state = dec.risk_state
                risk_budget = dec.risk_budget
            except ValueError as exc:
                if "NaN" in str(exc):
                    risk_state = "ON"
                    risk_budget = 1.0
                else:
                    raise
        gated_exposure = apply_risk_gating(desired_exposure, risk_state, risk_budget)
        desired.append(desired_exposure)
        gated.append(gated_exposure)
        prev_exp = exposures[-1] if exposures else 0.0
        delta = gated_exposure - prev_exp
        price_now = prices.iloc[i]
        fee = abs(delta) * price_now * fee_rate
        slip = abs(delta) * price_now * (slippage_bps / 10000.0)
        fee_paid += fee + slip
        if abs(delta) > 0:
            trades += 1
        turnover += abs(delta)
        next_eq = equity[-1] * (1 + gated_exposure * returns.iloc[i]) - fee - slip
        equity.append(next_eq)
        exposures.append(gated_exposure)
    equity_series = pd.Series(equity[1:], index=prices.index[1:])
    exposure_series = pd.Series(exposures, index=prices.index[:-1])
    final_return = float(equity_series.iloc[-1] / initial_equity - 1.0) if not equity_series.empty else 0.0
    n_hours = max(len(equity_series), 1)
    cagr = (equity_series.iloc[-1] / initial_equity) ** (ANNUAL_HOURS / n_hours) - 1 if n_hours > 0 else 0.0
    ret_series = equity_series.pct_change().dropna()
    realized_vol = float(ret_series.std(ddof=0) * np.sqrt(ANNUAL_HOURS)) if not ret_series.empty else 0.0
    sharpe = float(ret_series.mean() * ANNUAL_HOURS / (ret_series.std(ddof=0) + 1e-12)) if not ret_series.empty else 0.0
    best, worst = _best_worst_windows(equity_series, window_bars)
    avg_trade_size = float(turnover / trades) if trades else 0.0
    metrics = {
        "final_return_pct": final_return * 100.0,
        "cagr": cagr,
        "max_drawdown": _max_drawdown(equity_series),
        "realized_vol": realized_vol,
        "sharpe": sharpe,
        "trades_count": float(trades),
        "turnover": turnover,
        "time_in_market": float(np.mean(np.abs(exposure_series))) if not exposure_series.empty else 0.0,
        "average_exposure": float(np.mean(np.abs(exposure_series))) if not exposure_series.empty else 0.0,
        "fee_paid_estimate": fee_paid,
        "avg_trade_size": avg_trade_size,
    }
    return BacktestResult(
        equity=equity_series,
        exposure=exposure_series,
        trades=trades,
        turnover=turnover,
        fee_paid=fee_paid,
        metrics=metrics,
        best_windows=best,
        worst_windows=worst,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark strategies across symbols and psi modes.")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--psi-modes", default=",".join(DEFAULT_PSI_MODES))
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--limit-total", type=int, default=8000)
    ap.add_argument("--workdir", default="bench_out")
    ap.add_argument("--out", default="bench_out/strategies.csv")
    ap.add_argument("--pivot-out", default="")
    ap.add_argument("--plots-dir", default="")
    ap.add_argument("--fee-rate", type=float, default=0.001)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--max-exposure", type=float, default=1.0)
    ap.add_argument("--initial-equity", type=float, default=1000.0)
    ap.add_argument("--window-days", type=float, default=30.0)
    ap.add_argument("--window-bars", type=int, default=0)
    ap.add_argument("--kalman-mode", choices=["meanrev", "trend"], default="meanrev")
    ap.add_argument("--use-cache-only", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    psi_modes = [m.strip() for m in args.psi_modes.split(",") if m.strip()]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir) if args.plots_dir else None
    if plots_dir:
        plots_dir.mkdir(parents=True, exist_ok=True)

    window_bars = int(args.window_bars) if args.window_bars > 0 else int(args.window_days * 24) if args.window_days > 0 else 0
    rows: List[Dict] = []
    window_rows: List[Dict] = []

    for symbol in symbols:
        ohlcv, cache_path = load_or_fetch(symbol, args.timeframe, args.limit_total, workdir, args.use_cache_only)
        if ohlcv.empty:
            continue
        for psi_mode in psi_modes:
            cfg = FeatureConfig(psi_mode=psi_mode)
            for strategy_name in ("meanrev", "kalman"):
                res = run_backtest(
                    ohlcv=ohlcv,
                    strategy_name=strategy_name,
                    psi_mode=psi_mode,
                    feature_cfg=cfg,
                    fee_rate=args.fee_rate,
                    slippage_bps=args.slippage_bps,
                    max_exposure=args.max_exposure,
                    initial_equity=args.initial_equity,
                    window_bars=window_bars,
                    kalman_mode=args.kalman_mode,
                )
                params = {
                    "strategy": strategy_name,
                    "psi_mode": psi_mode,
                    "fee_rate": args.fee_rate,
                    "slippage_bps": args.slippage_bps,
                    "max_exposure": args.max_exposure,
                }
                rows.append(
                    {
                        "symbol": symbol,
                        "psi_mode": psi_mode,
                        "strategy_variant": strategy_name,
                        "variant": strategy_name,
                        "params_hash": params_hash(params),
                        "final_return": res.metrics.get("final_return_pct", 0.0) / 100.0,
                        "trades": res.metrics.get("trades_count", 0.0),
                        **res.metrics,
                    }
                )
                win_path = workdir / f"windows_{symbol.replace('/','_')}_{psi_mode}_{strategy_name}.json"
                with win_path.open("w") as f:
                    json.dump({"best": res.best_windows, "worst": res.worst_windows}, f, default=str)
                for w in res.best_windows:
                    window_rows.append(
                        {
                            "symbol": symbol,
                            "psi_mode": psi_mode,
                            "variant": strategy_name,
                            "type": "best",
                            "window_start": w.get("start"),
                            "window_end": w.get("end"),
                            "return": w.get("return"),
                            "max_drawdown": w.get("max_drawdown"),
                        }
                    )
                for w in res.worst_windows:
                    window_rows.append(
                        {
                            "symbol": symbol,
                            "psi_mode": psi_mode,
                            "variant": strategy_name,
                            "type": "worst",
                            "window_start": w.get("start"),
                            "window_end": w.get("end"),
                            "return": w.get("return"),
                            "max_drawdown": w.get("max_drawdown"),
                        }
                    )
                if plots_dir and HAS_MPL:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(res.equity.index, res.equity.values)
                    ax.set_title(f"{symbol} {strategy_name} {psi_mode}")
                    plt.tight_layout()
                    fig.savefig(plots_dir / f"equity_{symbol.replace('/','_')}_{psi_mode}_{strategy_name}.png")
                    plt.close(fig)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path}")
    if window_rows:
        windows_path = workdir / "windows.csv"
        pd.DataFrame(window_rows).to_csv(windows_path, index=False)
        print(f"Saved windows to {windows_path}")
    if args.pivot_out:
        pivot_cols = [c for c in ("final_return_pct", "max_drawdown") if c in summary_df.columns]
        if pivot_cols:
            pivot = summary_df.pivot_table(index="symbol", columns="strategy_variant", values=pivot_cols)
            pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
            pivot.to_csv(args.pivot_out, index=True)


if __name__ == "__main__":
    main()
