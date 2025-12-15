import pandas as pd
import pytest

from theta_bot_averaging.backtest import run_backtest


def test_predicted_return_required():
    idx = pd.date_range("2024-01-01", periods=2, freq="H")
    df = pd.DataFrame({"future_return": [0.01, -0.01]}, index=idx)
    position = pd.Series([0, 1], index=idx)
    with pytest.raises(ValueError):
        run_backtest(df, position=position)
