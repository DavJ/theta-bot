import pandas as pd
import pytest

from theta_bot_averaging.backtest import run_backtest


def test_transaction_costs_applied():
    idx = pd.date_range("2024-01-01", periods=3, freq="H")
    df = pd.DataFrame(
        {
            "future_return": [0.01, 0.0, -0.01],
            "predicted_return": [0.01, 0.0, -0.01],
        },
        index=idx,
    )
    position = pd.Series([0, 1, 1], index=idx, dtype=float)
    res = run_backtest(
        df,
        position=position,
        fee_rate=0.001,
        slippage_bps=0.0,
        spread_bps=0.0,
    )
    trades = res.trades
    # Entry at idx1 should incur cost = fee_rate
    assert trades.loc[idx[1], "costs"] == pytest.approx(0.001)
    # Net return at idx1 should be gross (0) - cost
    assert trades.loc[idx[1], "net_return"] == pytest.approx(-0.001)
