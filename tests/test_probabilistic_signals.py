"""Analytic posterior checks, causal clocks and bounded optional signal fusion."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from spot_bot.portfolio.probabilistic import (
    ProbabilisticConfig, bayesian_prediction, conditional_monte_carlo,
    conservative_fusion, probabilistic_signal_policies, walk_forward_probabilistic)


def small_config(**kwargs):
    return ProbabilisticConfig(minimum_training=30, training_window=60,
                               analogue_count=16, simulations=256, **kwargs)


def history(n=110):
    rng = np.random.default_rng(875)
    index = pd.date_range('2024-01-01', periods=n, freq='D', tz='UTC')
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(.001, .02, n))), index=index)
    features = pd.DataFrame({'r1': close.pct_change(), 'state': np.sin(np.arange(n) / 7)}, index=index)
    return features, close


def test_intercept_only_posterior_matches_scalar_conjugate_solution():
    y = np.array([.01, .02, -.03, .04, .005])
    prediction = bayesian_prediction(np.zeros((len(y), 1)), y, np.zeros(1))
    expected_mean = y.sum() / (len(y) + 1)
    shape = 3 + len(y) / 2
    scale = .01 + .5 * (y @ y - y.sum()**2 / (len(y) + 1))
    assert prediction['mean'] == pytest.approx(expected_mean)
    assert prediction['variance'] == pytest.approx(scale / (shape - 1) * (1 + 1 / (len(y) + 1)))
    assert prediction['latent_variance'] < prediction['variance']
    assert prediction['df'] == 2 * shape


def test_posterior_uncertainty_increases_for_extrapolated_feature_state():
    x = np.linspace(-1, 1, 50)[:, None]
    y = .02 * x[:, 0] + np.sin(np.arange(50)) * .001
    inside = bayesian_prediction(x, y, [0.])
    outside = bayesian_prediction(x, y, [10.])
    assert outside['latent_variance'] > inside['latent_variance']
    assert outside['variance'] > inside['variance']


def test_monte_carlo_preserves_exact_constant_return_blocks():
    x = np.arange(32, dtype=float)[:, None]
    result = conditional_monte_carlo(x, np.full(32, .012), [20.], np.random.default_rng(3),
                                    {'nominal': .0032, 'stress': .02}, small_config())
    assert result['mean'] == pytest.approx(.012)
    assert result['q05'] == result['q95'] == .012
    assert result['prob_nominal'] == 1
    assert result['prob_stress'] == 0
    assert result['analogue_count'] == 16


def test_monte_carlo_ties_are_chronological_and_integration_error_is_numerical():
    config = small_config()
    x, y = np.zeros((32, 1)), np.arange(32) / 100
    result = conditional_monte_carlo(x, y, [0.], np.random.default_rng(22), {'nominal': .07}, config)
    assert result['neighbors'].tolist() == list(range(16))
    assert result['exact_analogue_mean'] == pytest.approx(y[:16].mean())
    assert abs(result['mean'] - result['exact_analogue_mean']) < 5 * result['numerical_mean_se']
    larger = conditional_monte_carlo(x, y, [0.], np.random.default_rng(22), {'nominal': .07},
                                    replace(config, simulations=1024))
    assert larger['analogue_count'] == result['analogue_count']
    assert larger['numerical_mean_se'] / result['numerical_mean_se'] == pytest.approx(.5, rel=.1)


def test_duplicate_forecasts_do_not_create_predictive_precision():
    errors = np.tile(np.array([.03, .03]), (30, 1))
    fused = conservative_fusion([.01, .01], [.0009, .0009], errors)
    assert fused['mean'] == pytest.approx(.01)
    assert fused['variance'] >= .0009
    assert fused['weight_bayes'] == pytest.approx(.5)


def test_completed_error_pairs_reduce_weight_of_bad_expert_without_negative_weights():
    errors = np.column_stack((np.full(30, .001), np.full(30, .1)))
    result = conservative_fusion([.01, .05], [.001, .001], errors)
    assert .5 < result['weight_bayes'] <= 1
    assert 0 <= result['weight_mc'] < .5
    assert result['weight_bayes'] + result['weight_mc'] == pytest.approx(1)


def test_future_prices_features_and_longer_prefix_cannot_change_any_past_prediction():
    features, close = history()
    config, hurdles = small_config(), {'nominal': .0032, 'stress': .0062}
    prefix = walk_forward_probabilistic(features.iloc[:85], close.iloc[:85], symbol='BTCUSDT',
                                       hurdles=hurdles, config=config)
    features.iloc[85:] *= 1000
    close.iloc[85:] *= 4
    full = walk_forward_probabilistic(features, close, symbol='BTCUSDT', hurdles=hurdles, config=config)
    pd.testing.assert_frame_equal(prefix, full.iloc[:85])
    ready = full.loc[full.ready]
    for column in ('last_training_outcome_at', 'last_analogue_outcome_at', 'last_fusion_outcome_at'):
        assert ready[column].dropna().le(ready.loc[ready[column].notna(), 'available_at']).all()


def test_first_fit_waits_for_matured_labels_and_bayes_uses_nonoverlapping_blocks():
    features, close = history(60)
    features.r1 = 0.
    result = walk_forward_probabilistic(features, close, symbol='ETHUSDT', hurdles={'nominal': .0032},
                                       config=small_config())
    assert not result.ready.iloc[:34].any()
    assert result.ready.iloc[34]
    assert result.training_count.iloc[34] == 30
    assert result.bayes_observations.iloc[34] == 6
    assert result.last_training_outcome_at.iloc[34] == result.available_at.iloc[34]


def test_seed_changes_only_mc_source_and_result_is_reproducible_after_other_calls():
    features, close = history(60)
    config, hurdles = small_config(), {'nominal': .0032}
    first = walk_forward_probabilistic(features, close, symbol='BTCUSDT', hurdles=hurdles, config=config)
    walk_forward_probabilistic(features, close, symbol='ETHUSDT', hurdles=hurdles, config=config)
    repeat = walk_forward_probabilistic(features, close, symbol='BTCUSDT', hurdles=hurdles, config=config)
    other = walk_forward_probabilistic(features, close, symbol='BTCUSDT', hurdles=hurdles,
                                      config=replace(config, seed=config.seed + 1))
    pd.testing.assert_frame_equal(first, repeat)
    np.testing.assert_allclose(first.bayes_mean, other.bayes_mean, equal_nan=True)
    assert (first.loc[first.ready, 'mc_mean'] != other.loc[other.ready, 'mc_mean']).any()


def test_missing_current_feature_abstains_and_calendar_gap_is_rejected():
    features, close = history(60)
    features.iloc[50] = np.nan
    result = walk_forward_probabilistic(features, close, symbol='BTCUSDT', hurdles={'nominal': .0032},
                                       config=small_config())
    assert not result.ready.iloc[50]
    assert result.ready.iloc[49] and result.ready.iloc[51]
    with pytest.raises(ValueError, match='daily'):
        walk_forward_probabilistic(features.drop(features.index[40]), close.drop(close.index[40]),
                                  symbol='BTCUSDT', hurdles={'nominal': .0032}, config=small_config())


def forecast_inputs():
    index = pd.date_range('2024-01-01', periods=3, freq='4h', tz='UTC')
    index = pd.date_range(index[0], periods=18, freq='4h', tz='UTC')
    base = pd.DataFrame(.25, index=index, columns=['BTCUSDT', 'ETHUSDT'])
    daily = pd.date_range(index[0], periods=3, freq='D', tz='UTC')
    f = pd.DataFrame({'ready': [True, True, False]}, index=daily)
    for expert in ('bayes', 'mc', 'fused'):
        f[expert + '_mean'] = [.02, -.01, np.nan]
        f[expert + '_prob_nominal'] = [.6, .2, np.nan]
    return base, {'ETHUSDT': f.copy(), 'BTCUSDT': f.copy()}


def test_disabled_integration_does_not_read_invalid_forecasts():
    base, _ = forecast_inputs()
    result = probabilistic_signal_policies(base, object())
    assert list(result) == ['spot_control']
    pd.testing.assert_frame_equal(result['spot_control'][0], base)
    assert result['spot_control'][1] is None


def test_models_publish_only_at_completed_close_and_cannot_increase_base_weight():
    base, forecasts = forecast_inputs()
    policies = probabilistic_signal_policies(base, forecasts, enabled=True, hurdle=.0032)
    assert len(policies) == 8
    for name, (target, buy_mask) in policies.items():
        assert list(target.columns) == list(base.columns)
        assert (target <= base + 1e-12).all().all()
        assert (target >= 0).all().all()
        assert target.sum(axis=1).max() <= .5 + 1e-12
        if name != 'spot_control':
            assert target.iloc[:5].eq(0).all().all()
        if buy_mask is not None:
            assert list(buy_mask.columns) == list(base.columns)
            assert not buy_mask.iloc[:5].any().any()
            assert buy_mask.iloc[5].all()
            assert not buy_mask.iloc[11].any()
    assert policies['bayes_blend'][0].iloc[11].eq(.125).all()
    assert policies['ready_control'][0].iloc[-1].eq(0).all()


@pytest.mark.parametrize('override', [{'horizon': 0}, {'seed': -1}, {'simulations': 10},
                                     {'minimum_training': True}, {'prior_shape': 1},
                                     {'slope_precision': np.inf}])
def test_invalid_configuration_rejects(override):
    with pytest.raises(ValueError):
        ProbabilisticConfig(**override)
