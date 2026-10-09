"""Completed-data causality, weekly ranks and owned-position buy restrictions."""
import numpy as np
import pandas as pd
import pytest

from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.multiscale import MULTISCALE_VARIANTS, MultiscaleSpot, multiscale_targets, _weekly_top_two
from spot_bot.portfolio.momentum_refinement import MomentumRefinement


def markets(days=150):
    rng = np.random.default_rng(793)
    dates = pd.date_range("2024-01-01", periods=days * 6, freq="4h", tz="UTC").as_unit("ns")
    result = {}
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        close = 100 * np.exp(np.cumsum(rng.normal(.001, .018, len(dates))))
        open_ = np.r_[100., close[:-1]]
        result[symbol] = pd.DataFrame({"open": open_, "close": close,
            "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * .99,
            "volume": 100.}, index=dates)
    return result


@pytest.mark.parametrize("name", MULTISCALE_VARIANTS)
def test_future_and_partial_day_do_not_change_targets_permissions_fills_or_nav(name):
    data = markets()
    stop = 107 * 6 + 2
    prefix = {s: f.iloc[:stop] for s, f in data.items()}
    full, full_mask, _ = multiscale_targets(data)
    past, past_mask, _ = multiscale_targets(prefix)
    pd.testing.assert_frame_equal(past[name], full[name].iloc[:stop])
    if full_mask[name] is not None:
        pd.testing.assert_frame_equal(past_mask[name], full_mask[name].iloc[:stop])
    assert full[name].iloc[:60 * 6].eq(0).all().all()
    if name == "dual_horizon":
        assert full[name].iloc[:90 * 6].eq(0).all().all()
    assert full[name].min().min() >= 0 and full[name].max().max() <= .25
    assert full[name].sum(axis=1).max() <= .5 + 1e-12
    eq_p, fills_p, _ = run_intraday_spot(prefix, past[name], daily_control=True, completed_buy_mask=past_mask[name])
    eq_f, fills_f, _ = run_intraday_spot(data, full[name], daily_control=True, completed_buy_mask=full_mask[name])
    pd.testing.assert_frame_equal(eq_p, eq_f.iloc[:stop])
    pd.testing.assert_frame_equal(fills_p,
        fills_f.loc[fills_f.timestamp < data["BTCUSDT"].index[stop]].reset_index(drop=True))
    assert reconcile_spot_path(data, eq_f, fills_f)["reconciled"]


def test_both_controls_keep_exact_preceding_targets():
    data = markets()
    actual, _, daily = multiscale_targets(data)
    close = pd.DataFrame({s: f.close for s, f in daily.items()})
    for name, old_name in (("spot_control", "spot_control"), ("capped_control", "capped_allocation")):
        expected = MomentumRefinement(old_name).weights(close)
        expected.index = expected.index + pd.Timedelta("20h")
        expected = expected.reindex(actual[name].index, method="ffill", fill_value=0.)
        pd.testing.assert_frame_equal(actual[name], expected)


def test_all_true_buy_mask_is_identical_to_the_unchanged_default_replay():
    data = markets(110)
    targets, _, _ = multiscale_targets(data)
    mask = targets["spot_control"].astype(bool) | True
    a, f_a, s_a = run_intraday_spot(data, targets["spot_control"], daily_control=True)
    b, f_b, s_b = run_intraday_spot(data, targets["spot_control"], daily_control=True, completed_buy_mask=mask)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(f_a, f_b)
    assert s_a == s_b


def test_weekly_rank_does_not_chase_midweek_winners_or_replace_a_daily_exit():
    dates = pd.date_range("2024-01-01", periods=16, freq="D", tz="UTC")
    score = pd.DataFrame([[3., 2., 1.]] * len(dates), index=dates, columns=["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    active = pd.DataFrame(True, index=dates, columns=score.columns)
    score.loc[dates >= pd.Timestamp("2024-01-10", tz="UTC"), "BNBUSDT"] = 4.
    active.loc[dates >= pd.Timestamp("2024-01-11", tz="UTC"), "ETHUSDT"] = False
    membership = _weekly_top_two(score, active)
    assert membership.iloc[:6].eq(False).all().all()
    assert membership.loc["2024-01-08"].tolist() == [True, True, False]
    assert membership.loc["2024-01-10"].tolist() == [True, True, False]
    assert membership.loc["2024-01-11"].tolist() == [True, False, False]
    assert membership.loc["2024-01-15"].tolist() == [True, False, True]
    tied = pd.DataFrame(1., index=dates, columns=score.columns)
    assert _weekly_top_two(tied, tied.gt(0)).loc["2024-01-08"].tolist() == [True, True, False]


def flat_market(n=12):
    dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    frame = pd.DataFrame(100., index=dates, columns=["open", "high", "low", "close", "volume"])
    return {"BTCUSDT": frame}, pd.DataFrame(.25, index=dates, columns=["BTCUSDT"])


@pytest.mark.parametrize("delay,buy_bar,sell_bar", [(0, 3, 6), (1, 4, 7)])
def test_buy_permission_is_lagged_and_blocks_additions_to_actual_inventory(delay, buy_bar, sell_bar):
    data, target = flat_market()
    frame = data["BTCUSDT"]
    mask = pd.DataFrame(False, index=target.index, columns=target.columns)
    mask.iloc[2] = True
    # Lower prices create a genuine desired addition to an existing position.
    frame.iloc[5:, frame.columns.get_indexer(["open", "high", "low", "close"])] = 80.
    target.iloc[5:] = 0.
    eq, fills, _ = run_intraday_spot(data, target, completed_buy_mask=mask, signal_delay_bars=delay,
        fee_rate=0., slippage_bps=0., spread_bps=0., min_notional=0.)
    assert fills.timestamp.tolist() == [frame.index[buy_bar], frame.index[sell_bar]]
    assert fills.side.tolist() == ["buy", "sell"]
    assert eq.base_BTCUSDT.iloc[5] == pytest.approx(2.5)
    assert reconcile_spot_path(data, eq, fills)["reconciled"]


def test_false_buy_permission_does_not_prevent_cap_sale_or_inactivity_exit():
    data, target = flat_market()
    frame = data["BTCUSDT"]
    frame.iloc[3:, frame.columns.get_indexer(["open", "high", "low", "close"])] = 200.
    target.iloc[3:] = 0.
    mask = pd.DataFrame(False, index=target.index, columns=target.columns)
    mask.iloc[0] = True
    eq, fills, _ = run_intraday_spot(data, target, completed_buy_mask=mask,
        fee_rate=0., slippage_bps=0., spread_bps=0.)
    assert fills.timestamp.tolist() == [frame.index[1], frame.index[3], frame.index[4]]
    assert fills.side.tolist() == ["buy", "sell", "sell"]
    assert eq.base_BTCUSDT.iloc[4] == 0.
    assert reconcile_spot_path(data, eq, fills)["reconciled"]


def test_overheated_price_still_has_signal_but_cannot_authorize_a_buy():
    close = pd.DataFrame(100., index=pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"), columns=["BTCUSDT"])
    close.iloc[70:] = 140.
    policy = MultiscaleSpot("capped_no_chase")
    assert policy.weights(close).iloc[70, 0] == .25
    assert not policy.buy_permissions(close).iloc[70, 0]
    assert policy.buy_permissions(close).iloc[80, 0]


@pytest.mark.parametrize("kind", ["float", "nan", "nullable", "columns", "timestamps"])
def test_invalid_buy_permissions_are_rejected(kind):
    data, target = flat_market()
    mask = pd.DataFrame(True, index=target.index, columns=target.columns)
    if kind == "float":
        mask = mask.astype(float)
    elif kind == "nan":
        mask = mask.astype(object)
        mask.iloc[0, 0] = np.nan
    elif kind == "nullable":
        mask = mask.astype("boolean")
        mask.iloc[0, 0] = pd.NA
    elif kind == "columns":
        mask.columns = ["UNOWNED"]
    else:
        mask.index = mask.index + pd.Timedelta("4h")
    with pytest.raises(ValueError, match="buy mask"):
        run_intraday_spot(data, target, completed_buy_mask=mask)


@pytest.mark.parametrize("kwargs", [{"variant": "post_result_tuning"}, {"variant": "consensus", "max_exposure": np.inf},
    {"variant": "consensus", "asset_cap": -.1}, {"variant": "consensus", "rebalance_weekday": 1},
    {"variant": "consensus", "rebalance_band": -.1}])
def test_invalid_multiscale_config_rejected(kwargs):
    with pytest.raises(ValueError):
        MultiscaleSpot(**kwargs)
