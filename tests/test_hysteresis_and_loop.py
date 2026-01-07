import pandas as pd
import pytest

from spot_bot import run_live
from spot_bot.core import legacy_adapter
from spot_bot.features import FeatureConfig
from spot_bot.regime.types import RegimeDecision
from spot_bot.strategies.base import Intent


class _AlwaysOnRegime:
    def decide(self, features_df):
        return RegimeDecision(risk_state="ON", risk_budget=1.0, reason="always-on", diagnostics={})


class _FixedStrategy:
    def __init__(self, exposure: float, zscore: float = -1.0):
        self.exposure = exposure
        self.zscore = zscore

    def generate_intent(self, features_df):
        return Intent(desired_exposure=self.exposure, reason="fixed", diagnostics={"zscore": self.zscore})


def _stub_features(ohlcv_df):
    idx = pd.to_datetime(ohlcv_df.index, utc=True)
    data = {
        "S": pd.Series(0.5, index=idx),
        "C": pd.Series(0.5, index=idx),
        "C_int": pd.Series(0.1, index=idx),
        "rv": pd.Series(0.05, index=idx),
        "psi": pd.Series(0.0, index=idx),
    }
    return pd.DataFrame(data, index=idx)


def _sample_ohlcv(rows: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = pd.Series(20000.0, index=idx)
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 10,
            "low": base - 10,
            "close": base,
            "volume": 1.0,
        },
        index=idx,
    )


def test_hysteresis_blocks_small_exposure_change(monkeypatch):
    df = _sample_ohlcv()
    monkeypatch.setattr(legacy_adapter, "compute_features", lambda ohlcv, cfg: _stub_features(ohlcv))
    regime = _AlwaysOnRegime()
    strategy = _FixedStrategy(0.1, zscore=-0.2)
    balances = {"usdt": 1000.0, "btc": 0.005}

    result = run_live.compute_step(
        ohlcv_df=df,
        feature_cfg=FeatureConfig(),
        regime_engine=regime,
        strategy=strategy,
        max_exposure=1.0,
        fee_rate=0.001,
        balances=balances,
        mode="dryrun",
    )

    assert result.delta_btc == pytest.approx(0.0)
    current_exposure = balances["btc"] * result.close / result.equity["equity_usdt"]
    assert result.target_exposure == pytest.approx(current_exposure)


def test_hysteresis_allows_large_exposure_change(monkeypatch):
    df = _sample_ohlcv()
    monkeypatch.setattr(legacy_adapter, "compute_features", lambda ohlcv, cfg: _stub_features(ohlcv))
    regime = _AlwaysOnRegime()
    strategy = _FixedStrategy(0.6, zscore=-3.0)
    balances = {"usdt": 1000.0, "btc": 0.0}

    result = run_live.compute_step(
        ohlcv_df=df,
        feature_cfg=FeatureConfig(),
        regime_engine=regime,
        strategy=strategy,
        max_exposure=1.0,
        fee_rate=0.001,
        balances=balances,
        mode="dryrun",
    )

    assert abs(result.delta_btc) > 0
    assert result.target_exposure > 0.05


def test_loop_state_skips_when_no_new_bar(tmp_path, monkeypatch):
    df = _sample_ohlcv()
    monkeypatch.setattr(legacy_adapter, "compute_features", lambda ohlcv, cfg: _stub_features(ohlcv))
    store_path = tmp_path / "loop_state.json"
    state_store = run_live.LoopStateStore(path=store_path)
    regime = _AlwaysOnRegime()
    strategy = _FixedStrategy(0.3)
    balances = {"usdt": 1000.0, "btc": 0.0}

    df_trade, bar_row, ts_value, bar_state = run_live._prepare_trade_data(df, "1h", "bar_close")
    assert bar_state == "closed"
    assert ts_value.tzinfo is not None
    ts_ms = run_live._to_epoch_ms(ts_value)

    result = run_live.compute_step(
        ohlcv_df=df_trade,
        feature_cfg=FeatureConfig(),
        regime_engine=regime,
        strategy=strategy,
        max_exposure=1.0,
        fee_rate=0.001,
        balances=balances,
        mode="dryrun",
    )
    assert result is not None

    state_store.save_last_closed_ts(ts_ms)
    assert state_store.load_last_closed_ts() == ts_ms
    assert store_path.exists()

    df_trade_again, _, ts_value_again, _ = run_live._prepare_trade_data(df, "1h", "bar_close")
    should_skip = state_store.load_last_closed_ts() == run_live._to_epoch_ms(ts_value_again)
    assert should_skip
