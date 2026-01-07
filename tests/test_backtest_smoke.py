import sys

import numpy as np
import pandas as pd

from spot_bot.backtest import run_backtest

WARMUP_BUFFER = 10


def _synthetic_ohlcv(rows: int) -> pd.DataFrame:
    np.random.seed(0)
    idx = pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC")
    base = 20000 + np.linspace(0, 50, rows)
    noise = np.sin(np.linspace(0, 20, rows)) * 20
    close = base + noise
    open_ = close * (1 + np.random.normal(0, 0.0005, size=rows))
    spread = np.abs(np.random.normal(0, 0.0008, size=rows))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = np.full(rows, 1.0)
    return pd.DataFrame(
        {"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_backtest_smoke_kalman_dual():
    rows = 2000
    df = _synthetic_ohlcv(rows)
    psi_window = 512
    rv_window = 120
    conc_window = 256
    equity_df, trades_df, summary = run_backtest(
        df=df,
        timeframe="1m",
        strategy_name="kalman_mr_dual",
        psi_mode="scale_phase",
        psi_window=psi_window,
        rv_window=rv_window,
        conc_window=conc_window,
        base=10.0,
        fee_rate=0.001,
        slippage_bps=5.0,
        max_exposure=0.3,
    )
    warmup = psi_window + rv_window + conc_window
    assert len(equity_df) <= rows
    assert len(equity_df) >= rows - warmup - WARMUP_BUFFER
    assert "final_equity" in summary
    assert "sharpe" in summary
    assert "maxDD" in summary
    assert not equity_df.empty
    assert summary["final_equity"] > 0


def test_backtest_without_pyarrow(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    df = _synthetic_ohlcv(300)
    equity_df, trades_df, summary = run_backtest(
        df=df,
        timeframe="1m",
        strategy_name="meanrev",
        psi_mode="scale_phase",
        psi_window=128,
        rv_window=60,
        conc_window=128,
        base=10.0,
        fee_rate=0.0005,
        slippage_bps=0.5,
        max_exposure=0.25,
    )
    assert not equity_df.empty
    assert "final_equity" in summary
