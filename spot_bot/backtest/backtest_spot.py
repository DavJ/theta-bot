from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import argparse

import numpy as np
import pandas as pd

from spot_bot.features import FeatureConfig, compute_features
from spot_bot.portfolio import compute_target_position
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.base import Strategy
from spot_bot.strategies.mean_reversion import MeanReversionStrategy

if TYPE_CHECKING:
    from spot_bot.persist import SQLiteLogger


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.Series
    metrics: Dict[str, float]
    risk_state: Optional[pd.Series] = None
    exposure: Optional[pd.Series] = None


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    safe_peak = peak.where(peak > 0, 1e-8)
    drawdown = (equity_curve - peak) / safe_peak
    return float(drawdown.min())


def _run_backtest_core(
    ohlcv_df: pd.DataFrame,
    strategy: Strategy,
    regime_engine: Optional[RegimeEngine],
    regime_features: Optional[pd.DataFrame],
    fee_rate: float,
    max_exposure: float,
    initial_equity: float,
    logger: "SQLiteLogger | None" = None,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    if "close" not in ohlcv_df.columns:
        raise ValueError("ohlcv_df must contain a 'close' column.")
    prices = ohlcv_df["close"].astype(float)
    if len(prices) < 2:
        empty_series = pd.Series(dtype=float)
        return BacktestResult(
            equity_curve=empty_series,
            positions=empty_series,
            metrics={
                "final_return": 0.0,
                "max_drawdown": 0.0,
                "time_in_market": 0.0,
                "turnover": 0.0,
                "trades": 0.0,
                "avg_trade_size": 0.0,
            },
            risk_state=None,
            exposure=None,
        )

    equity = float(initial_equity)
    position = 0.0
    equity_curve = []
    position_history = []
    exposure_history = []
    timestamps = []
    turnover_value = 0.0
    trades = 0
    risk_history = []

    for i in range(len(prices) - 1):
        window_df = ohlcv_df.iloc[: i + 1]
        intent = strategy.generate_intent(window_df[["close"]])

        risk_state = "ON"
        risk_budget = 1.0
        if regime_engine is not None:
            if regime_features is None:
                raise ValueError(
                    "regime_features must be provided when regime_engine is set "
                    "(use compute_features or pass feature_config to run_strategy_backtests)."
                )
            features_slice = regime_features.iloc[: i + 1]
            latest_feat = features_slice.iloc[-1]
            if not (pd.isna(latest_feat.get("S")) or pd.isna(latest_feat.get("C"))):
                decision = regime_engine.decide(features_slice)
                risk_state = decision.risk_state
                risk_budget = decision.risk_budget

        target_position = compute_target_position(
            equity_usdt=equity,
            price=prices.iloc[i],
            desired_exposure=intent.desired_exposure,
            risk_budget=risk_budget,
            max_exposure=max_exposure,
            risk_state=risk_state,
        )

        trade = target_position - position
        turnover_value += abs(trade) * prices.iloc[i]
        fee = abs(trade) * prices.iloc[i] * fee_rate
        slippage_cost = abs(trade) * prices.iloc[i] * (slippage_bps / 10000.0)
        equity -= fee + slippage_cost
        if abs(trade) > 0:
            trades += 1
        if logger and trade != 0:
            logger.log_execution(
                timestamp=prices.index[i],
                action="buy" if trade > 0 else "sell",
                qty=abs(trade),
                price=float(prices.iloc[i]),
                fee=fee + slippage_cost,
                order_id=f"bt_{i}",
                status="filled",
            )

        exposure_fraction = 0.0
        if equity > 0:
            exposure_fraction = min(1.0, abs(target_position) * prices.iloc[i] / equity)
        exposure_history.append(exposure_fraction)
        risk_history.append(risk_state)

        position = target_position
        next_price = prices.iloc[i + 1]
        pnl = position * (next_price - prices.iloc[i])
        equity += pnl

        equity_curve.append(equity)
        position_history.append(position)
        timestamps.append(prices.index[i + 1])
        if logger:
            usdt_balance = equity - position * next_price
            logger.log_equity(timestamp=prices.index[i + 1], equity_usdt=equity, btc=position, usdt=usdt_balance)

    equity_series = pd.Series(equity_curve, index=timestamps)
    position_series = pd.Series(position_history, index=timestamps)
    exposure_series = pd.Series(exposure_history, index=timestamps)
    risk_series = pd.Series(risk_history, index=timestamps) if risk_history else None
    time_in_market = float(np.mean(exposure_history)) if exposure_history else 0.0
    turnover = turnover_value / initial_equity if initial_equity != 0.0 else 0.0

    metrics = {
        "final_return": float(equity_series.iloc[-1] / initial_equity - 1.0)
        if (not equity_series.empty and initial_equity != 0.0)
        else 0.0,
        "max_drawdown": _max_drawdown(equity_series),
        "time_in_market": time_in_market,
        "turnover": float(turnover),
        "trades": float(trades),
        "avg_trade_size": float(turnover_value / trades) if trades else 0.0,
    }

    return BacktestResult(
        equity_curve=equity_series, positions=position_series, metrics=metrics, risk_state=risk_series, exposure=exposure_series
    )


def run_strategy_backtests(
    ohlcv_df: pd.DataFrame,
    strategy: Strategy,
    fee_rate: float = 0.0005,
    max_exposure: float = 1.0,
    initial_equity: float = 1000.0,
    regime_config: Optional[Dict] = None,
    feature_config: Optional[FeatureConfig] = None,
    logger: "SQLiteLogger | None" = None,
    slippage_bps: float = 0.0,
) -> Dict[str, BacktestResult]:
    """
    Run strategy backtests with and without risk gating.

    Returns a mapping with 'baseline' (no gating) and 'gated' (RegimeEngine).
    """
    baseline = _run_backtest_core(
        ohlcv_df=ohlcv_df,
        strategy=strategy,
        regime_engine=None,
        regime_features=None,
        fee_rate=fee_rate,
        max_exposure=max_exposure,
        initial_equity=initial_equity,
        logger=logger,
        slippage_bps=slippage_bps,
    )

    feat_cfg = feature_config or FeatureConfig()
    regime_features = compute_features(ohlcv_df, cfg=feat_cfg)
    regime_engine = RegimeEngine(regime_config or {})
    gated = _run_backtest_core(
        ohlcv_df=ohlcv_df,
        strategy=strategy,
        regime_engine=regime_engine,
        regime_features=regime_features,
        fee_rate=fee_rate,
        max_exposure=max_exposure,
        initial_equity=initial_equity,
        logger=logger,
        slippage_bps=slippage_bps,
    )

    return {"baseline": baseline, "gated": gated}


def run_mean_reversion_backtests(
    ohlcv_df: pd.DataFrame,
    fee_rate: float = 0.0005,
    max_exposure: float = 1.0,
    initial_equity: float = 1000.0,
    regime_config: Optional[Dict] = None,
    strategy: Optional[MeanReversionStrategy] = None,
    feature_config: Optional[FeatureConfig] = None,
    logger: "SQLiteLogger | None" = None,
    slippage_bps: float = 0.0,
) -> Dict[str, BacktestResult]:
    """
    Backward-compatible wrapper that defaults to MeanReversionStrategy.
    """
    mr_strategy = strategy or MeanReversionStrategy()
    return run_strategy_backtests(
        ohlcv_df=ohlcv_df,
        strategy=mr_strategy,
        fee_rate=fee_rate,
        max_exposure=max_exposure,
        initial_equity=initial_equity,
        regime_config=regime_config,
        feature_config=feature_config,
        logger=logger,
        slippage_bps=slippage_bps,
    )


def _load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp")
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {missing}")
    return df[["open", "high", "low", "close", "volume"]]


def _build_feature_config_from_args(args: argparse.Namespace) -> FeatureConfig:
    return FeatureConfig(
        base=args.base,
        rv_window=args.rv_window,
        conc_window=args.conc_window,
        psi_window=args.psi_window,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run spot bot backtest with regime gating.")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV with timestamp column.")
    parser.add_argument("--fee-rate", type=float, default=0.0005, help="Transaction fee rate per trade.")
    parser.add_argument("--max-exposure", type=float, default=1.0, help="Maximum exposure fraction.")
    parser.add_argument("--initial-equity", type=float, default=1000.0, help="Starting equity in quote currency.")
    parser.add_argument("--s-off", dest="s_off", type=float, default=-0.1, help="Score threshold for OFF.")
    parser.add_argument("--s-on", dest="s_on", type=float, default=0.2, help="Score threshold for ON/REDUCE.")
    parser.add_argument("--rv-off", dest="rv_off", type=float, default=0.05, help="Vol threshold for OFF.")
    parser.add_argument("--rv-reduce", dest="rv_reduce", type=float, default=0.04, help="Vol threshold for REDUCE.")
    parser.add_argument("--base", type=float, default=FeatureConfig.base, help="Log-phase base (default 10).")
    parser.add_argument("--rv-window", type=int, default=FeatureConfig.rv_window, help="RV window.")
    parser.add_argument(
        "--conc-window", type=int, default=FeatureConfig.conc_window, help="Rolling window for concentration."
    )
    parser.add_argument("--psi-window", type=int, default=FeatureConfig.psi_window, help="Rolling window for psi.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ohlcv = _load_ohlcv_csv(args.csv)
    regime_config = {
        "s_off": args.s_off,
        "s_on": args.s_on,
        "rv_off": args.rv_off,
        "rv_reduce": args.rv_reduce,
    }
    feat_cfg = _build_feature_config_from_args(args)
    results = run_mean_reversion_backtests(
        ohlcv_df=ohlcv,
        fee_rate=args.fee_rate,
        max_exposure=args.max_exposure,
        initial_equity=args.initial_equity,
        regime_config=regime_config,
        feature_config=feat_cfg,
    )
    for name, res in results.items():
        print(
            f"[{name}] final_return={res.metrics['final_return']:.4f} "
            f"max_drawdown={res.metrics['max_drawdown']:.4f} "
            f"time_in_market={res.metrics['time_in_market']:.4f}"
        )


if __name__ == "__main__":
    main()
