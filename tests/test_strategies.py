import pandas as pd

from bench.benchmark_strategies import run_backtest, _best_worst_windows
from spot_bot.features import FeatureConfig
from spot_bot.strategies import KalmanRiskStrategy, MeanRevGatedStrategy, apply_risk_gating


def _synthetic_ohlcv(n=50):
    idx = pd.date_range("2024-01-01", periods=n, freq="H")
    close = pd.Series(100 + 0.1 * pd.RangeIndex(n), index=idx)
    open_ = close * 0.999
    high = close * 1.001
    low = close * 0.999
    vol = pd.Series(1.0, index=idx)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_strategy_meanrev_gating():
    strat = MeanRevGatedStrategy(max_exposure=1.0)
    prices = pd.Series([100, 99, 98, 97, 96, 95, 94, 93, 92, 91])
    out = strat.generate(prices)
    gated = apply_risk_gating(out.desired_exposure, "REDUCE", 0.5)
    assert abs(gated) <= abs(out.desired_exposure)
    assert gated == out.desired_exposure * 0.5


def test_kalman_shapes():
    strat = KalmanRiskStrategy(mode="meanrev", min_bars=5)
    prices = pd.Series([100, 100.2, 100.4, 100.3, 100.1, 100.0, 99.9, 99.8])
    out = strat.generate(prices)
    assert abs(out.desired_exposure) <= strat.max_exposure
    assert all(pd.notna(list(out.diagnostics.values())))
    assert out.diagnostics["innovation_var"] > 0


def test_backtest_fee_application():
    ohlcv = _synthetic_ohlcv(60)
    res = run_backtest(
        ohlcv=ohlcv,
        strategy_name="meanrev",
        psi_mode="none",
        feature_cfg=FeatureConfig(psi_mode="none"),
        fee_rate=0.01,
        slippage_bps=5,
        max_exposure=1.0,
        initial_equity=1000.0,
        window_bars=24,
        kalman_mode="meanrev",
    )
    assert res.fee_paid > 0
    assert res.equity.iloc[-1] < 1000.0
    best, worst = _best_worst_windows(res.equity, 24)
    assert isinstance(best, list) and isinstance(worst, list)
