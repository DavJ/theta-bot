"""Causal flow signals, clock handling and independently reconciled spot cash."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.download_spot_flow import parse_flow_archive, validate_bars
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path, rolling_gains
from spot_bot.portfolio.flow import spot_flow_targets, daily_control_targets, flow_features


def markets(n=500):
    rng = np.random.default_rng(76)
    dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    result = {}
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        close = 100 * np.exp(np.cumsum(rng.normal(.001, .015, n)))
        open_ = np.r_[100, close[:-1]]
        volume = rng.uniform(50, 200, n)
        buy = rng.uniform(.35, .65, n) * volume
        result[symbol] = pd.DataFrame({"open": open_, "close": close,
            "high": np.maximum(open_, close) * 1.005, "low": np.minimum(open_, close) * .995,
            "volume": volume, "quote_volume": volume * close, "trade_count": 100,
            "taker_buy_base": buy, "taker_buy_quote": buy * close}, index=dates)
    return result


@pytest.mark.parametrize("name", ["price_momentum", "flow_momentum", "price_breakout", "flow_breakout", "flow_reversion", "flow_blend"])
def test_future_bars_do_not_change_past_flow_targets_fills_or_equity(name):
    data = markets()
    short = {s: f.iloc[:413] for s, f in data.items()}
    a = spot_flow_targets(short)[name]
    b = spot_flow_targets(data)[name]
    pd.testing.assert_frame_equal(a, b.iloc[:413])
    assert (b >= 0).all().all() and b.max().max() <= .25 and b.sum(axis=1).max() <= .5 + 1e-12
    eq_a, t_a, _ = run_intraday_spot(short, a)
    eq_b, t_b, _ = run_intraday_spot(data, b)
    pd.testing.assert_frame_equal(eq_a, eq_b.iloc[:413])
    pd.testing.assert_frame_equal(t_a, t_b.loc[t_b.timestamp < data["BTCUSDT"].index[413]].reset_index(drop=True))
    assert reconcile_spot_path(data, eq_b, t_b)["reconciled"]


def test_completed_target_and_flow_can_only_execute_on_the_following_open():
    data = {"BTCUSDT": markets(10)["BTCUSDT"]}
    frame = data["BTCUSDT"]
    frame[["open", "close", "high", "low"]] = 100.
    target = pd.DataFrame(.25, index=frame.index, columns=list(data))
    target.iloc[2:] = 0
    eq, fills, summary = run_intraday_spot(data, target)
    assert list(fills.timestamp) == [frame.index[1], frame.index[3]]
    assert list(fills.side) == ["buy", "sell"]
    assert summary["net_pnl"] == pytest.approx(-summary["fees_paid_total"] - summary["slippage_paid_total"])
    assert summary["fees_paid_total"] == pytest.approx(2.5 * (100.06 + 99.94) * .001)
    assert summary["slippage_paid_total"] == pytest.approx(.3)
    assert reconcile_spot_path(data, eq, fills)["reconciled"]
    delayed, delayed_fills, _ = run_intraday_spot(data, target, signal_delay_bars=1)
    assert list(delayed_fills.timestamp) == [frame.index[2], frame.index[4]]


def test_extreme_spot_crash_preserves_nonnegative_funds_and_inventory():
    data = {"BTCUSDT": markets(10)["BTCUSDT"]}
    frame = data["BTCUSDT"]
    frame[["open", "close", "high", "low"]] = 100.
    frame.iloc[5:, frame.columns.get_indexer(["open", "close", "high", "low"])] = .0001
    target = pd.DataFrame(1., index=frame.index, columns=list(data))
    eq, fills, summary = run_intraday_spot(data, target, max_exposure=1, asset_cap=1, fee_rate=.01)
    assert eq.usdt.min() >= 0
    assert eq.base_BTCUSDT.min() >= 0
    assert summary["total_return"] < -.99
    assert reconcile_spot_path(data, eq, fills)["reconciled"]


@pytest.mark.parametrize("invalid", ["negative", "borrowed", "nan", "gap"])
def test_invalid_targets_or_missing_market_intervals_are_rejected(invalid):
    data = markets(10)
    target = pd.DataFrame(.1, index=data["BTCUSDT"].index, columns=list(data))
    if invalid == "gap":
        data["ETHUSDT"] = data["ETHUSDT"].iloc[:-1]
    else:
        target.iloc[0, 0] = {"negative": -.1, "borrowed": 1.1, "nan": np.nan}[invalid]
    with pytest.raises(ValueError):
        run_intraday_spot(data, target)


def test_flow_standardization_uses_only_prior_baseline():
    frame = markets()["BTCUSDT"]
    features = flow_features(frame)
    pressure = ((2 * frame.taker_buy_base - frame.volume).rolling(3).sum() / frame.volume.rolling(3).sum())
    t = 220
    history = pressure.iloc[t-180:t]
    expected = (pressure.iloc[t] - history.mean()) / max(.001, history.std(ddof=0))
    assert features.flow_z.iloc[t] == pytest.approx(expected)


@pytest.mark.parametrize("month,unit", [("2024-02", "ms"), ("2025-02", "us")])
def test_archive_timestamps_and_taker_columns_across_unit_change(month, unit):
    begin = pd.Timestamp(month + "-01", tz="UTC")
    index = pd.date_range(begin, begin + pd.offsets.MonthBegin(1), freq="4h", inclusive="left")
    stamps = index.as_unit(unit).astype("int64")
    close_stamps = (index + pd.Timedelta("4h") - pd.Timedelta(1, unit=unit)).as_unit(unit).astype("int64")
    raw = pd.DataFrame({0: stamps, 1: 100, 2: 101, 3: 99, 4: 100, 5: 10,
                        6: close_stamps, 7: 1000, 8: 20, 9: 6, 10: 600, 11: 0})
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(f"BTCUSDT-4h-{month}.csv", raw.to_csv(index=False, header=False))
    result = parse_flow_archive(payload.getvalue(), "BTCUSDT", month)
    assert pd.DatetimeIndex(result.timestamp).equals(index)
    assert result.taker_buy_base.eq(6).all() and result.trade_count.eq(20).all()


@pytest.mark.parametrize("field,value", [("taker_buy_base", 1e9), ("quote_volume", -1), ("trade_count", .5), ("taker_buy_quote", 1e9)])
def test_bad_executed_volume_is_rejected(field, value):
    frame = markets(10)["BTCUSDT"]
    frame[field] = frame[field].astype(float)
    frame.iloc[0, frame.columns.get_loc(field)] = value
    with pytest.raises(ValueError, match="Invalid spot"):
        validate_bars(frame)


def test_rolling_windows_include_initial_capital_and_are_not_period_resets():
    daily = pd.Series(1000 * 2 ** (np.arange(1, 101) / 30),
        index=pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"))
    result = rolling_gains(daily, 1000)
    assert result["7"]["windows"] == 94
    assert result["7"]["doublings"] == 0
    assert result["30"]["maximum_return"] == pytest.approx(1)
    assert result["90"]["maximum_return"] == pytest.approx(7)


def test_independent_ledger_detects_corrupted_equity():
    data = markets(20)
    target = pd.DataFrame(.1, index=data["BTCUSDT"].index, columns=list(data))
    eq, fills, _ = run_intraday_spot(data, target)
    eq.loc[7, "equity"] += .01
    with pytest.raises(ValueError, match="reconcile"):
        reconcile_spot_path(data, eq, fills)


def test_rising_close_does_not_create_a_retroactive_open_drawdown():
    data = {"BTCUSDT": markets(3)["BTCUSDT"]}
    frame = data["BTCUSDT"]
    frame[["open", "close", "high", "low"]] = 100.
    frame.loc[frame.index[1], ["high", "close"]] = 200.
    frame.loc[frame.index[2], ["open", "high", "low", "close"]] = 200.
    target = pd.DataFrame(.25, index=frame.index, columns=list(data))
    eq, _, _ = run_intraday_spot(data, target, fee_rate=0, slippage_bps=0, spread_bps=0)
    assert eq.observed_drawdown.eq(0).all()


def test_daily_control_delayed_by_one_bar_trades_at_four_am():
    data = markets(6 * 100)
    target, _ = daily_control_targets(data)
    _, fills, _ = run_intraday_spot(data, target, daily_control=True, signal_delay_bars=1)
    assert not fills.empty
    assert fills.timestamp.dt.hour.eq(4).all()
