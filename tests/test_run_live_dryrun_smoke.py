import numpy as np
import pandas as pd
import pytest

from spot_bot.features import FeatureConfig
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.run_live import compute_step
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


def _build_synthetic_ohlcv(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="H")
    base = 20000 + np.linspace(0, 100, rows)
    close = base + np.sin(np.linspace(0, 6.28, rows)) * 50
    open_ = close * 0.999
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(rows, 1.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_compute_step_dryrun_smoke():
    df = _build_synthetic_ohlcv()
    feature_cfg = FeatureConfig(rv_window=24, conc_window=64, psi_window=64)
    engine = RegimeEngine({})
    strategy = MeanReversionStrategy()

    result = compute_step(
        ohlcv_df=df,
        feature_cfg=feature_cfg,
        regime_engine=engine,
        strategy=strategy,
        max_exposure=0.3,
        fee_rate=0.001,
        balances={"usdt": 1000.0, "btc": 0.0},
        mode="dryrun",
    )

    assert result.execution is None
    assert result.target_exposure >= 0.0
    expected_btc = (
        result.equity["equity_usdt"] * result.target_exposure / result.close if result.close > 0 else 0.0
    )
    assert result.target_btc == pytest.approx(expected_btc)
