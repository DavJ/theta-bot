import pandas as pd
import pytest

from theta_bot_averaging.data import build_targets


def test_future_return_alignment_no_leakage():
    idx = pd.date_range("2024-01-01", periods=5, freq="H")
    close = [100.0, 102.0, 101.0, 103.0, 104.0]
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )

    out = build_targets(df, horizon=2, threshold_bps=0)
    # Last 2 rows should be dropped due to lack of future data
    assert len(out) == 3
    expected_ret = (101.0 / 100.0) - 1.0
    assert out.iloc[0]["future_return"] == pytest.approx(expected_ret)
    # Alignment: label computed strictly forward
    assert out.iloc[-1]["future_return"] == pytest.approx((104.0 / 101.0) - 1.0)
