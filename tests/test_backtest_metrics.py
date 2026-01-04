import pandas as pd

from theta_bot_averaging.backtest import run_backtest


def test_trade_count_and_turnover():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    df = pd.DataFrame(
        {
            "future_return": [0.0, 0.0, 0.0, 0.0, 0.0],
            "predicted_return": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    position = pd.Series([0, 1, 1, 0, -1], index=idx, dtype=float)

    res = run_backtest(df, position=position, fee_rate=0.0, slippage_bps=0.0, spread_bps=0.0)

    assert res.metrics["trade_count"] == 3
    assert res.metrics["turnover"] == position.diff().abs().sum() / len(position)
