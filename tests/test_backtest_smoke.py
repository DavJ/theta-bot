import sys
import subprocess
import textwrap
from pathlib import Path

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


def test_backtest_without_pyarrow():
    # Pandas detects Arrow at import and may retain Arrow-backed string arrays.
    # Removing an already imported dependency corrupts that live environment;
    # block it before importing Pandas in a fresh interpreter instead.
    script = textwrap.dedent('''
        import importlib.abc
        import sys

        class NoArrow(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pyarrow" or fullname.startswith("pyarrow."):
                    raise ModuleNotFoundError("Arrow intentionally unavailable", name=fullname)

        sys.meta_path.insert(0, NoArrow())
        from tests.test_backtest_smoke import _synthetic_ohlcv, run_backtest

        equity, trades, summary = run_backtest(
            df=_synthetic_ohlcv(500), timeframe="1m", strategy_name="meanrev",
            psi_mode="scale_phase", psi_window=128, rv_window=60,
            conc_window=128, base=10.0, fee_rate=0.0005,
            slippage_bps=0.5, max_exposure=0.25,
        )
        assert not equity.empty
        assert summary["final_equity"] > 0
        assert "pyarrow" not in sys.modules
    ''')
    result = subprocess.run([sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
