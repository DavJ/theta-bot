import numpy as np
import pandas as pd

from spot_bot.features import FeatureConfig
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.run_live import run_replay
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


def _build_history(rows: int = 180) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = 20000 + np.linspace(0, 150, rows)
    oscillation = np.sin(np.linspace(0, 8, rows)) * 40
    close = base + oscillation
    open_ = close * 0.998
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = np.full(rows, 2.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_replay_smoke(tmp_path):
    df = _build_history()
    csv_path = tmp_path / "history.csv"
    df.reset_index().rename(columns={"index": "timestamp"}).to_csv(csv_path, index=False)

    feature_cfg = FeatureConfig(rv_window=24, conc_window=64, psi_window=64)
    regime_engine = RegimeEngine({})
    strategy = MeanReversionStrategy()

    equity_path = tmp_path / "equity_curve.csv"
    trades_path = tmp_path / "trades.csv"

    equity_df, trades_df, _ = run_replay(
        ohlcv_df=df,
        feature_cfg=feature_cfg,
        regime_engine=regime_engine,
        strategy=strategy,
        max_exposure=0.3,
        fee_rate=0.001,
        slippage_bps=1.0,
        spread_bps=0.5,
        hyst_k=5.0,
        hyst_floor=0.02,
        hyst_mode="exposure",
        min_notional=1.0,
        step_size=None,
        initial_usdt=1000.0,
        initial_btc=0.0,
        equity_path=equity_path,
        trades_path=trades_path,
        features_path=None,
    )

    assert not equity_df.empty
    assert (equity_df["equity_usdt"] > 0).all()
    assert len(trades_df) < len(equity_df)
    assert equity_path.exists()
    assert trades_path.exists()
