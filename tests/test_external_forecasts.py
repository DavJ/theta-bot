"""Publication-time joins and purged, training-only short-horizon forecasts."""
import numpy as np
import pandas as pd
import pytest

from scripts.download_external_signals import fred_availability
from spot_bot.features.external import asof_signal, distributed_lags, external_features, validate_signals
from spot_bot.portfolio.forecast import (ForecastConfig, forecast_targets, price_features,
                                         training_recency_weights, walk_forward_forecast)
from tests.test_execution_portfolio import markets


def signals(n=200):
    dates = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"source": "FEAR_GREED", "event_time": dates,
                         "available_at": dates + pd.Timedelta("1D"),
                         "value": 50 + 20 * np.sin(np.arange(n) / 10)})


def test_publication_delay_and_expiry_prevent_future_and_stale_values():
    dates = pd.date_range("2024-01-01", periods=8, tz="UTC")
    observations = pd.DataFrame({"event_time": [dates[0]], "available_at": [dates[2]], "value": [123.]})
    joined = asof_signal(observations, dates, pd.Timedelta("2D"))
    assert joined.value.iloc[:2].isna().all()
    assert joined.value.iloc[2:5].eq(123).all()
    assert joined.value.iloc[5:].isna().all()


def test_fred_weekend_and_holiday_do_not_create_early_publications():
    dates = pd.Series(pd.to_datetime(["2026-10-02", "2023-12-22"], utc=True))
    actual = fred_availability(dates, "DGS10")
    assert actual.iloc[0] == pd.Timestamp("2026-10-06", tz="UTC")
    assert actual.iloc[1] == pd.Timestamp("2023-12-27", tz="UTC")


def test_batch_availability_uses_latest_known_observation_without_early_release():
    index = pd.date_range("2022-10-07", periods=8, tz="UTC")
    observations = pd.DataFrame({"event_time": [index[3], index[0]],
                                 "available_at": [index[5], index[5]], "value": [11., 10.]})
    validated = validate_signals(observations.assign(source="NASDAQCOM"))
    joined = asof_signal(validated, index, pd.Timedelta("3D"))
    assert joined.value.iloc[:5].isna().all()
    assert joined.value.iloc[5:].eq(11).all()


@pytest.mark.parametrize("decision_unit,source_unit", [("us", "ns"), ("ns", "us")])
def test_asof_join_normalizes_timestamp_units_without_rounding_release_jitter(decision_unit, source_unit):
    dates = pd.date_range("2024-01-01", periods=6, tz="UTC")
    decisions = dates.astype(f"datetime64[{decision_unit}, UTC]")
    observations = pd.DataFrame({"event_time": [dates[0]],
        "available_at": [dates[2] + pd.Timedelta("1ms")], "value": [123.]})
    for column in ("event_time", "available_at"):
        observations[column] = observations[column].astype(f"datetime64[{source_unit}, UTC]")
    joined = asof_signal(observations, decisions, pd.Timedelta("4D"))
    assert joined.value.iloc[:3].isna().all()
    assert joined.value.iloc[3:].eq(123).all()
    assert joined.available_at.iloc[3] == dates[2] + pd.Timedelta("1ms")


@pytest.mark.parametrize("column", ["event_time", "available_at"])
def test_mixed_subsecond_provider_timestamps_preserve_timezone(column):
    frame = signals(10)
    frame.loc[1, column] += pd.Timedelta("1ms")
    if column == "event_time":
        frame.loc[1, "available_at"] += pd.Timedelta("1ms")
    frame[column] = frame[column].astype(str)
    validated = validate_signals(frame)
    expected = pd.Timestamp("2022-01-02", tz="UTC") + pd.Timedelta("1ms")
    if column == "available_at":
        expected += pd.Timedelta("1D")
    assert validated.loc[1, column] == expected


@pytest.mark.parametrize("fault", ["duplicate", "early", "naive", "value"])
def test_invalid_source_grain_times_and_values_are_rejected(fault):
    frame = signals(10)
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif fault == "early":
        frame.loc[0, "available_at"] -= pd.Timedelta("2D")
    elif fault == "naive":
        frame["event_time"] = frame.event_time.dt.tz_localize(None)
    else:
        frame.loc[0, "value"] = 101
    with pytest.raises(ValueError):
        validate_signals(frame)


def test_external_prefix_and_extra_delay_do_not_backfill_new_values():
    frame = signals()
    index = pd.DatetimeIndex(frame.event_time)
    full, _ = external_features(frame, index)
    prefix, _ = external_features(frame.iloc[:150], index[:150])
    pd.testing.assert_frame_equal(prefix["sentiment"], full["sentiment"].iloc[:150])
    delayed, _ = external_features(frame, index, extra_delay=pd.Timedelta("3D"))
    assert delayed["sentiment"].iloc[:3].isna().all().all()
    np.testing.assert_allclose(delayed["sentiment"].iloc[8:].to_numpy(),
                               full["sentiment"].iloc[5:-3].to_numpy())


def test_lag_buckets_cover_exact_past_windows_and_wait_for_missing_history():
    index = pd.date_range("2024-01-01", periods=50, tz="UTC")
    features = pd.DataFrame({"signal": np.arange(50, dtype=float)}, index=index)
    lagged = distributed_lags(features)
    assert lagged.loc[index[30], "signal_lag0_2"] == np.mean([28, 29, 30])
    assert lagged.loc[index[30], "signal_lag3_9"] == np.mean(np.arange(21, 28))
    assert lagged.loc[index[30], "signal_lag10_20"] == np.mean(np.arange(10, 21))
    assert lagged.signal_lag10_20.iloc[:20].isna().all()
    features.iloc[15] = np.nan
    missing = distributed_lags(features)
    assert np.isnan(missing.loc[index[30], "signal_lag10_20"])
    assert np.isfinite(missing.loc[index[30], "signal_lag0_2"])


def test_lag_buckets_reject_calendar_gaps_instead_of_changing_delay_units():
    index = pd.date_range("2024-01-01", periods=30, tz="UTC").delete(10)
    with pytest.raises(ValueError, match="daily grid"):
        distributed_lags(pd.DataFrame({"signal": 1.}, index=index))


def test_recency_weights_use_completed_outcome_age_and_fixed_half_life():
    config = ForecastConfig()
    rows = np.array([0, 126, 252])
    weights = training_recency_weights(rows, 257, config)
    assert weights.mean() == pytest.approx(1)
    assert weights[1] / weights[0] == pytest.approx(2)
    assert weights[2] / weights[1] == pytest.approx(2)
    with pytest.raises(ValueError, match="completed"):
        training_recency_weights([253], 257, config)


def test_ridge_scaler_and_regression_both_use_train_only_recency_weights():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    index = pd.date_range("2024-01-01", periods=60, tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(60)) / 100)), index=index)
    features = pd.DataFrame({"known": np.arange(60) ** 2}, index=index)
    config = ForecastConfig(minimum_training=20, training_window=30, recency_half_life_days=10)
    result = walk_forward_forecast(features, close, config)
    rows, current = np.arange(20), 24
    weights = training_recency_weights(rows, current, config)
    x = features.iloc[rows].to_numpy()
    y = (close.shift(-5) / close - 1).iloc[rows].to_numpy()
    y = np.clip(y, -3 * y.std(ddof=0), 3 * y.std(ddof=0))
    scaler = StandardScaler().fit(x, sample_weight=weights)
    model = Ridge(alpha=10).fit(scaler.transform(x), y, sample_weight=weights)
    expected = model.predict(scaler.transform(features.iloc[[current]].to_numpy()))[0]
    assert result.forecast.iloc[current] == pytest.approx(expected)
    assert result.effective_training_count.iloc[current] == pytest.approx(
        weights.sum() ** 2 / np.square(weights).sum())


@pytest.mark.parametrize("algorithm", ["ridge", "boost"])
def test_future_features_and_prices_cannot_change_past_predictions(algorithm):
    rng = np.random.default_rng(555)
    index = pd.date_range("2022-01-01", periods=230, tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(.001, .02, len(index)))), index=index)
    features = pd.DataFrame({"return": close.pct_change(), "wave": np.sin(np.arange(len(index)))}, index=index)
    config = ForecastConfig(algorithm, minimum_training=30, training_window=100)
    prefix = walk_forward_forecast(features.iloc[:190], close.iloc[:190], config)
    altered, altered_close = features.copy(), close.copy()
    altered.iloc[190:] *= 10000
    altered_close.iloc[190:] *= 2
    full = walk_forward_forecast(altered, altered_close, config)
    pd.testing.assert_frame_equal(prefix, full.iloc[:190])
    ready = full.forecast.notna()
    assert full.loc[ready, "last_training_outcome_at"].le(full.index[ready] + pd.Timedelta("1D")).all()


def test_first_fit_waits_for_completed_five_day_outcomes():
    index = pd.date_range("2024-01-01", periods=60, tz="UTC")
    close = pd.Series(np.exp(np.arange(60) / 100), index=index)
    features = pd.DataFrame({"known": np.sin(np.arange(60))}, index=index)
    config = ForecastConfig(minimum_training=20, training_window=30, horizon=5)
    result = walk_forward_forecast(features, close, config)
    assert result.forecast.iloc[:24].isna().all()
    assert result.training_count.iloc[24] == 20
    assert result.last_training_outcome_at.iloc[24] == index[24] + pd.Timedelta("1D")


def test_price_theta_features_use_only_completed_prefix_and_targets_obey_spot_limits():
    data = markets(300)
    full = price_features(data)
    prefix = price_features({s: f.iloc[:250] for s, f in data.items()})
    for symbol in data:
        pd.testing.assert_frame_equal(prefix[symbol], full[symbol].iloc[:250])
    close = pd.DataFrame({s: f.close for s, f in data.items()})
    forecast = close * 0 + 1
    targets = forecast_targets(forecast, close)
    assert targets.sum(axis=1).max() <= 1 + 1e-12
    assert targets.max().max() <= .5 + 1e-12
    assert targets.iloc[:60].eq(0).all().all()


def test_missing_features_abstain_instead_of_inventing_values():
    index = pd.date_range("2022-01-01", periods=100, tz="UTC")
    features = pd.DataFrame({"known": np.sin(np.arange(100))}, index=index)
    features.iloc[70] = np.nan
    close = pd.Series(100 * np.exp(np.arange(100) / 500), index=index)
    result = walk_forward_forecast(features, close, ForecastConfig(minimum_training=20))
    assert np.isnan(result.forecast.iloc[70])
    assert np.isfinite(result.forecast.iloc[69]) and np.isfinite(result.forecast.iloc[71])
