"""Causal daily entries, bounded allocation and exits independent of trading bands."""
import numpy as np
import pandas as pd
import pytest

from scripts.research_momentum_refinement import fill_frequency
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.flow import daily_control_targets
from spot_bot.portfolio.momentum_refinement import (
    MOMENTUM_VARIANTS, MomentumRefinement, momentum_refinement_targets, _redistribute_capped,
)
from spot_bot.portfolio.trend import TrendPortfolio


def markets(days=150):
    rng = np.random.default_rng(194)
    dates = pd.date_range("2024-01-01", periods=days * 6, freq="4h", tz="UTC").as_unit("ns")
    result = {}
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        close = 100 * np.exp(np.cumsum(rng.normal(.001, .018, len(dates))))
        open_ = np.r_[100., close[:-1]]
        result[symbol] = pd.DataFrame({"open": open_, "close": close,
            "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * .99,
            "volume": 100.}, index=dates)
    return result


@pytest.mark.parametrize("name", MOMENTUM_VARIANTS)
def test_future_prices_and_partial_day_do_not_change_past_targets_or_fills(name):
    data = markets()
    # Cut inside the next day: its partial close must not be published as a full day.
    stop = 113 * 6 + 3
    prefix = {s: frame.iloc[:stop] for s, frame in data.items()}
    targets, _ = momentum_refinement_targets(data)
    past, _ = momentum_refinement_targets(prefix)
    pd.testing.assert_frame_equal(past[name], targets[name].iloc[:stop])
    assert targets[name].iloc[:60 * 6].eq(0).all().all()
    assert (targets[name] >= 0).all().all()
    assert targets[name].max().max() <= .25
    assert targets[name].sum(axis=1).max() <= .5 + 1e-12
    band = MomentumRefinement(name).rebalance_band
    eq_p, fills_p, _ = run_intraday_spot(prefix, past[name], daily_control=True, rebalance_band=band)
    eq_f, fills_f, _ = run_intraday_spot(data, targets[name], daily_control=True, rebalance_band=band)
    pd.testing.assert_frame_equal(eq_p, eq_f.iloc[:stop])
    pd.testing.assert_frame_equal(fills_p,
        fills_f.loc[fills_f.timestamp < data["BTCUSDT"].index[stop]].reset_index(drop=True))
    assert reconcile_spot_path(data, eq_f, fills_f)["reconciled"]


def test_control_and_wider_band_have_identical_original_targets():
    data = markets()
    actual, daily = momentum_refinement_targets(data)
    expected, _ = daily_control_targets(data)
    pd.testing.assert_frame_equal(actual["spot_control"], expected)
    pd.testing.assert_frame_equal(actual["wide_band"], expected)
    close = pd.DataFrame({s: f.close for s, f in daily.items()})
    pd.testing.assert_frame_equal(MomentumRefinement("spot_control").weights(close),
        TrendPortfolio("momentum_vol", max_exposure=.5, asset_cap=.25).weights(close))


def test_confirmation_waits_for_three_positive_days_and_exits_without_confirmation():
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    close = pd.DataFrame(100., index=dates, columns=["BTCUSDT"])
    close.iloc[65:70, 0] = [101., 102., 103., 101., 99.]
    weights = MomentumRefinement("confirmed_entry").weights(close).BTCUSDT
    assert list(weights.iloc[65:70]) == [0., 0., .25, .25, 0.]


def test_strength_requires_stronger_entry_but_keeps_positive_weak_momentum_after_entry():
    rng = np.random.default_rng(25)
    values = 100 * np.exp(np.cumsum(rng.normal(0., .025, 100)))
    for pos, gain in zip(range(69, 74), (-.01, .001, .20, .005, -.001)):
        values[pos] = values[pos - 30] * (1 + gain)
    close = pd.DataFrame({"BTCUSDT": values},
        index=pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"))
    weights = MomentumRefinement("strong_entry").weights(close).BTCUSDT
    assert list(weights.iloc[69:74]) == [0., 0., .25, .25, 0.]
    pd.testing.assert_frame_equal(MomentumRefinement("strong_entry").weights(close),
        MomentumRefinement("strong_entry_wide").weights(close))


def test_btc_regime_blocks_altcoins_when_btc_is_flat_or_negative():
    close = pd.DataFrame(100., index=pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
                         columns=["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    close.iloc[70:73] = [[99., 105., 105.], [100., 105., 105.], [101., 105., 105.]]
    weights = MomentumRefinement("btc_regime").weights(close)
    assert weights.iloc[70:72].eq(0).all().all()
    assert weights.iloc[72].gt(0).all()
    assert MomentumRefinement("spot_control").weights(close).iloc[70].sum() == pytest.approx(.5)


@pytest.mark.parametrize("score,expected", [
    ([0., 0., 0.], [0., 0., 0.]), ([4., 0., 0.], [.25, 0., 0.]),
    ([9., 1., 0.], [.25, .25, 0.]), ([6., 3., 1.], [.25, .1875, .0625]),
    ([1., 1., 1.], [1 / 6, 1 / 6, 1 / 6]),
])
def test_capped_allocation_preserves_budget_and_cash_when_uninvestable(score, expected):
    actual = _redistribute_capped(np.array(score), .5, .25)
    assert actual == pytest.approx(expected)
    assert actual.sum() <= .5 + 1e-12 and actual.max() <= .25


def test_wider_band_skips_small_monday_trade_but_never_blocks_inactive_or_cap_sales():
    dates = pd.date_range("2023-12-31", periods=10 * 6, freq="4h", tz="UTC")
    frame = pd.DataFrame(100., index=dates, columns=["open", "high", "low", "close", "volume"])
    # On Wednesday the holding exceeds 25%, despite a change smaller than the 3pt band.
    frame.loc[dates >= pd.Timestamp("2024-01-03", tz="UTC"), ["open", "high", "low", "close"]] = 110.
    # Return below the cap before Monday so the ordinary trade has no cap override.
    frame.loc[dates >= pd.Timestamp("2024-01-08", tz="UTC"), ["open", "high", "low", "close"]] = 100.
    target = pd.DataFrame(.25, index=dates, columns=["BTCUSDT"])
    target.loc[dates >= pd.Timestamp("2024-01-07 20:00", tz="UTC")] = .21
    target.loc[dates >= pd.Timestamp("2024-01-08 20:00", tz="UTC")] = 0.
    results = {}
    for band in (.01, .03):
        eq, fills, _ = run_intraday_spot({"BTCUSDT": frame}, target, daily_control=True, rebalance_band=band)
        assert reconcile_spot_path({"BTCUSDT": frame}, eq, fills)["reconciled"]
        results[band] = fills
        assert fills.loc[fills.timestamp == pd.Timestamp("2024-01-03", tz="UTC"), "side"].tolist() == ["sell"]
        assert fills.loc[fills.timestamp == pd.Timestamp("2024-01-09", tz="UTC"), "side"].tolist() == ["sell"]
    monday = pd.Timestamp("2024-01-08", tz="UTC")
    assert len(results[.01].loc[results[.01].timestamp == monday]) == 1
    assert results[.03].loc[results[.03].timestamp == monday].empty


def test_fill_frequency_includes_zero_trade_months_and_counts_fills_not_round_trips():
    equity = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", "2024-02-29 20:00", freq="4h", tz="UTC")})
    fills = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-05T00:00:00Z"] * 3)})
    stats = fill_frequency(equity, fills)
    assert stats["calendar_days"] == 60 and stats["active_trading_days"] == 1
    assert stats["fills_per_month"] == 1.5
    assert stats["minimum_monthly_fills"] == 0 and stats["maximum_monthly_fills"] == 3
    empty = fill_frequency(equity, fills.iloc[:0])
    assert empty["fills_per_week"] == 0 and empty["active_trading_days"] == 0


@pytest.mark.parametrize("kwargs", [{"variant": "tune_after_results"},
    {"variant": "spot_control", "max_exposure": 1.1}, {"variant": "spot_control", "asset_cap": -.1},
    {"variant": "spot_control", "max_exposure": np.nan}])
def test_invalid_refinement_and_spot_caps_rejected(kwargs):
    with pytest.raises(ValueError):
        MomentumRefinement(**kwargs)


def test_missing_btc_and_missing_daily_information_are_rejected():
    close = pd.DataFrame(100., index=pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"), columns=["ETHUSDT"])
    with pytest.raises(ValueError, match="BTC"):
        MomentumRefinement("btc_regime").weights(close)
    with pytest.raises(ValueError, match="daily"):
        MomentumRefinement("confirmed_entry").weights(close.drop(close.index[70]))
