import pandas as pd
import pytest

from bench.benchmark_pairs import compute_equity_curve, compute_equity_metrics, max_drawdown


def _df(close, exposure):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="h")
    return (
        pd.DataFrame(
            {
                "timestamp": idx,
                "close": close,
                "target_exposure": exposure,
            }
        ).set_index("timestamp")
    )


def test_equity_curve_with_costs():
    df = _df([100.0, 110.0, 100.0], [0.0, 1.0, 1.0])
    equity, turnover, exp_used = compute_equity_curve(
        df,
        df["target_exposure"],
        fee_rate=0.001,
        slippage_bps=0.0,
        max_exposure=1.0,
    )
    assert pytest.approx(turnover, rel=1e-9) == 1.0
    # 10% gain on second bar minus 0.1% cost on the exposure change, then -9.09% move
    assert pytest.approx(equity.iloc[-1], rel=1e-6) == 0.90818181818
    assert len(exp_used) == len(equity)


def test_max_drawdown_calculation():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    equity = pd.Series([1.0, 1.2, 1.0, 1.5, 1.1], index=idx)
    assert max_drawdown(equity) == pytest.approx(-0.2666666667)


def test_no_lookahead_shift():
    df = _df([100.0, 110.0, 121.0], [1.0, 0.0, 0.0])
    equity, turnover, exp_used = compute_equity_curve(
        df,
        df["target_exposure"],
        fee_rate=0.0,
        slippage_bps=0.0,
        max_exposure=1.0,
    )
    assert pytest.approx(equity.iloc[-1], rel=1e-9) == 1.1
    metrics = compute_equity_metrics(equity, exp_used, turnover, timeframe="1h")
    assert metrics["final_return"] > 0
