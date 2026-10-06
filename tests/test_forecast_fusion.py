"""Correlated measurements, numerical stability, and causal calibration."""
import numpy as np
import pandas as pd
import pytest

from spot_bot.strategies.forecast_fusion import ForecastKalman, calibrate_and_fuse


def test_precise_source_receives_more_gain_and_joseph_covariance_is_positive():
    fusion = ForecastKalman(variance=1.0, persistence=1.0)
    mean, variance, gain = fusion.update([0.01, -0.01], np.diag([1e-5, 1e-2]), 0)
    assert gain[0] > 100 * gain[1]
    assert mean > 0.009
    assert 0 < variance < 1e-5


def test_correlated_duplicate_forecasts_do_not_count_as_independent_votes():
    independent = ForecastKalman(variance=1.0, persistence=1.0)
    correlated = ForecastKalman(variance=1.0, persistence=1.0)
    independent.update(np.ones(5), np.eye(5), 0)
    r = 0.99 * np.ones((5, 5)) + 0.01 * np.eye(5)
    correlated.update(np.ones(5), r, 0)
    assert correlated.variance > 2.9 * independent.variance
    assert correlated.variance == pytest.approx(0.5, rel=0.01)


@pytest.mark.parametrize("covariance", [np.zeros((2, 2)), [[1, 2], [0, 1]], np.eye(3)])
def test_invalid_covariance_is_rejected(covariance):
    with pytest.raises(ValueError):
        ForecastKalman().update([0.01, 0.02], covariance, 0)


@pytest.mark.parametrize("n_sources", [1, 5])
def test_calibration_matches_prefix_and_does_not_see_future_outcomes(n_sources):
    rng = np.random.default_rng(10)
    scores = pd.DataFrame(rng.normal(size=(350, n_sources)), columns=list("abcde")[:n_sources])
    # A known predictive relationship with next-day labels, plus noise.
    returns = 0.001 * scores.a.shift(1).fillna(0).to_numpy() + rng.normal(0, 0.01, len(scores))
    close = pd.Series(100 * np.cumprod(1 + returns))
    prefix = calibrate_and_fuse(scores.iloc[:250], close.iloc[:250])
    full = calibrate_and_fuse(scores, close)
    pd.testing.assert_frame_equal(prefix, full.iloc[:250])
    assert prefix.fused_return.iloc[:126].isna().all()
    assert prefix.posterior_variance.dropna().gt(0).all()
    changed = close.copy()
    changed.iloc[250:] *= 2
    future_changed = calibrate_and_fuse(scores, changed)
    pd.testing.assert_frame_equal(full.iloc[:250], future_changed.iloc[:250])
