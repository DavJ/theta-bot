"""Economic invariants for the optional variance-calibrated filter."""
import numpy as np
import pandas as pd
import pytest

from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy
from spot_bot.backtest.fast_backtest import run_backtest


def features(n=400):
    t = np.arange(n)
    return pd.DataFrame({"close": 100 * np.exp(0.01 * np.sin(t / 10) + 0.0001 * t),
                         "C": 0.6, "psi": (t / 50) % 1, "rv": 0.004})


@pytest.mark.parametrize("snr_enabled", [False, True])
def test_currency_rescaling_does_not_change_log_vol_intents(snr_enabled):
    original = features()
    scaled = original.copy()
    scaled.close *= 1000
    strategy = MeanRevDualKalmanStrategy(price_space="log_vol", snr_enabled=snr_enabled)
    np.testing.assert_allclose(strategy.generate_series(original), strategy.generate_series(scaled), atol=1e-10)
    a, b = strategy.generate_intent(original), strategy.generate_intent(scaled)
    assert a.desired_exposure == pytest.approx(b.desired_exposure, abs=1e-10)


@pytest.mark.parametrize("snr_enabled", [False, True])
def test_series_and_single_intent_agree_and_future_does_not_change_prefix(snr_enabled):
    frame = features()
    frame["risk_budget"] = np.linspace(0.2, 0.8, len(frame))
    strategy = MeanRevDualKalmanStrategy(price_space="log_vol", snr_enabled=snr_enabled)
    prefix = strategy.generate_series(frame.iloc[:300])
    full = strategy.generate_series(frame)
    pd.testing.assert_series_equal(prefix, full.iloc[:300])
    assert prefix.iloc[-1] == pytest.approx(strategy.generate_intent(frame.iloc[:300]).desired_exposure)


def test_return_variance_uses_only_previous_observations():
    strategy = MeanRevDualKalmanStrategy(price_space="log_vol")
    close = features().close
    changed = close.copy()
    changed.iloc[-1] *= 2
    z = pd.Series(0.0, index=close.index)
    a = strategy._run_filters(close, z)
    b = strategy._run_filters(changed, z)
    assert a.innovation_var == b.innovation_var
    assert a.residual != b.residual


@pytest.mark.parametrize("bad", [0, -1, np.inf])
def test_invalid_log_prices_are_rejected(bad):
    frame = features()
    frame.loc[100, "close"] = bad
    with pytest.raises(ValueError, match="finite positive"):
        MeanRevDualKalmanStrategy(price_space="log_vol").generate_series(frame)


def test_calibrated_backtest_keeps_previous_equity_when_future_is_appended():
    frame = pd.read_csv("data/BTCUSDT_1H_real.csv.gz").iloc[:450]

    def run(df):
        return run_backtest(df, "1h", "kalman_mr_dual", "scale_phase", 20, 10, 20,
                            2, 0.001, 5, 0.3, log=False, dual_price_space="log_vol")

    prefix, trades, _ = run(frame.iloc[:300])
    full, full_trades, _ = run(frame)
    pd.testing.assert_frame_equal(prefix, full.iloc[:len(prefix)].reset_index(drop=True))
    if not trades.empty:
        cutoff = prefix.timestamp.iloc[-1]
        pd.testing.assert_frame_equal(trades, full_trades.loc[full_trades.timestamp <= cutoff].reset_index(drop=True))
