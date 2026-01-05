import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_synthetic_ohlcv(path: Path, bars: int = 50) -> None:
    idx = pd.date_range("2024-01-01", periods=bars, freq="H")
    close = 100 + pd.Series(range(bars), index=idx) * 0.1 + pd.Series([0.5] * bars, index=idx)
    open_ = close.shift(1, fill_value=close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.999
    volume = pd.Series(1.0, index=idx)
    df = pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    df.to_csv(path, index=False)


def test_benchmark_strategies_smoke(tmp_path):
    workdir = tmp_path / "bench_out"
    workdir.mkdir()
    cached = workdir / "ohlcv_TEST_USDT.csv"
    _write_synthetic_ohlcv(cached, bars=48)

    summary_path = workdir / "summary.csv"
    pivot_path = workdir / "pivot.csv"

    cmd = [
        sys.executable,
        "bench/benchmark_strategies.py",
        "--symbols",
        "TEST/USDT",
        "--timeframe",
        "1h",
        "--limit-total",
        "48",
        "--workdir",
        str(workdir),
        "--out",
        str(summary_path),
        "--pivot-out",
        str(pivot_path),
        "--window-bars",
        "12",
        "--use-cache-only",
    ]
    res = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True)
    assert res.returncode == 0

    summary = pd.read_csv(summary_path)
    required = {"symbol", "variant", "final_return", "max_drawdown", "time_in_market", "turnover", "trades", "avg_trade_size"}
    assert required.issubset(set(summary.columns))
    assert not summary.empty

    pivot = pd.read_csv(pivot_path)
    assert not pivot.empty

    windows_file = workdir / "windows.csv"
    assert windows_file.exists()
    windows_df = pd.read_csv(windows_file)
    assert {"symbol", "variant", "window_start", "window_end"}.issubset(set(windows_df.columns))
