from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def _annualization_factor(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    delta = (index[1:] - index[:-1]).median()
    seconds = delta.total_seconds()
    if seconds == 0:
        return 1.0
    periods_per_year = (365 * 24 * 3600) / seconds
    return periods_per_year


def compute_metrics(returns: pd.Series, equity: pd.Series) -> Dict[str, float]:
    ann_factor = _annualization_factor(returns.index)
    total_return = equity.iloc[-1] - 1.0
    mean = returns.mean()
    std = returns.std()
    downside = returns[returns < 0].std()

    sharpe = (mean * ann_factor**0.5) / std if std and std > 0 else 0.0
    sortino = (mean * ann_factor**0.5) / downside if downside and downside > 0 else 0.0

    positive = returns[returns > 0].sum()
    negative = returns[returns < 0].sum()
    profit_factor = positive / abs(negative) if negative != 0 else np.inf

    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()

    return {
        "total_return": float(total_return),
        "cagr": float((equity.iloc[-1]) ** (ann_factor / len(returns)) - 1.0),
        "max_drawdown": float(_max_drawdown(equity)),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win) if not np.isnan(avg_win) else 0.0,
        "avg_loss": float(avg_loss) if not np.isnan(avg_loss) else 0.0,
    }
