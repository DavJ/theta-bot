import numpy as np
import pandas as pd

from spot_bot.backtest.backtest_spot import run_mean_reversion_backtests


def test_mean_reversion_backtest_smoke():
    bars = 40
    idx = pd.date_range("2024-01-01", periods=bars, freq="h")
    base = 20000 + np.linspace(0, 200, bars)
    close = base + np.sin(np.linspace(0, 3.14, bars)) * 20
    ohlcv = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
        index=idx,
    )

    results = run_mean_reversion_backtests(ohlcv, fee_rate=0.0, initial_equity=1000.0)

    assert set(results.keys()) == {"baseline", "gated"}
    for res in results.values():
        assert "final_return" in res.metrics
        assert "max_drawdown" in res.metrics
        assert "time_in_market" in res.metrics
        assert 0.0 <= res.metrics["time_in_market"] <= 1.0
