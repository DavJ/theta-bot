import pandas as pd

from spot_bot.run_live import latest_closed_ohlcv


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp("2024-01-01T00:00:00Z"), periods=3, freq="1h")
    data = {
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [1.0, 1.0, 1.0],
    }
    return pd.DataFrame(data, index=idx)


def test_latest_closed_ohlcv_drops_in_progress_bar():
    df = _sample_df()
    now = pd.Timestamp("2024-01-01T02:30:00Z")
    truncated = latest_closed_ohlcv(df, "1h", now=now)

    assert len(truncated) == 2
    assert truncated.index[-1] == df.index[1]


def test_latest_closed_ohlcv_keeps_closed_bar():
    df = _sample_df()
    now = pd.Timestamp("2024-01-01T03:05:00Z")
    truncated = latest_closed_ohlcv(df, "1h", now=now)

    assert len(truncated) == 3
    assert truncated.index[-1] == df.index[-1]
