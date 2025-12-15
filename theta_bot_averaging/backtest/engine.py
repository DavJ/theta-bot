from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .metrics import compute_metrics


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]


def _apply_transaction_costs(position: pd.Series, fee_rate: float, slippage_bps: float, spread_bps: float) -> pd.Series:
    # Costs applied on position changes (entries/exits)
    trades = position.diff().abs()
    fee_cost = trades * fee_rate
    slippage_cost = trades * (slippage_bps / 10_000.0)
    spread_cost = trades * (spread_bps / 10_000.0)
    return fee_cost + slippage_cost + spread_cost


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    future_return_col: str = "future_return",
    fee_rate: float = 0.0004,
    slippage_bps: float = 1.0,
    spread_bps: float = 0.5,
    output_dir: Optional[Path] = None,
) -> BacktestResult:
    if "predicted_return" not in df.columns:
        raise ValueError("Missing predicted_return. Run inference first or fix pipeline.")
    if future_return_col not in df.columns:
        raise ValueError("Missing predicted returns: future_return column not found.")

    if position.isna().any():
        position = position.fillna(0.0)

    costs = _apply_transaction_costs(position, fee_rate, slippage_bps, spread_bps)
    gross_returns = position.shift(1).fillna(0.0) * df[future_return_col]
    net_returns = gross_returns - costs

    equity = (1 + net_returns).cumprod()
    metrics = compute_metrics(net_returns, equity)
    trades_abs = position.diff().abs().fillna(0.0)
    metrics["trade_count"] = int(trades_abs.sum())
    metrics["turnover"] = float(trades_abs.sum() / len(position)) if len(position) else 0.0

    trades = pd.DataFrame(
        {
            "position": position,
            "future_return": df[future_return_col],
            "gross_return": gross_returns,
            "costs": costs,
            "net_return": net_returns,
        },
        index=df.index,
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output_dir / "trades.csv")
        equity.to_frame("equity").to_csv(output_dir / "equity_curve.csv")

    equity_curve = equity.to_frame("equity")
    return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)
