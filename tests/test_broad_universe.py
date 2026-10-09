"""Historical identity, missing-market accounting and completed-day selection clocks."""
import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.download_spot_universe import COHORT, COLUMNS, parse_partial_archive, assemble_asset, source_periods, fetch_month
from scripts.research_broad_universe import freeze_broad, paired_block_interval
from spot_bot.backtest.intraday import run_intraday_spot, reconcile_spot_path
from spot_bot.portfolio.broad_universe import (BROAD_VARIANTS, BroadSpot, broad_targets,
    complete_daily, liquidity_eligibility, weekly_top, reduce_volatility)
from spot_bot.portfolio.momentum_refinement import MomentumRefinement


def markets(days=160, start="2024-01-01"):
    rng = np.random.default_rng(9328)
    dates = pd.date_range(start, periods=days * 6, freq="4h", tz="UTC").as_unit("ns")
    result = {}
    for symbol in COHORT:
        close = 100 * np.exp(np.cumsum(rng.normal(.001, .012, len(dates))))
        open_ = np.r_[100., close[:-1]]
        result[symbol] = pd.DataFrame({"open": open_, "close": close,
            "high": np.maximum(open_, close) * 1.01, "low": np.minimum(open_, close) * .99,
            "volume": 1_000., "quote_volume": 20_000_000.}, index=dates)
    result["TERRA_CLASSIC"].iloc[30 * 6:70 * 6] = np.nan
    return result


@pytest.mark.parametrize("name", BROAD_VARIANTS)
def test_future_quotes_partial_days_and_gaps_do_not_change_past_targets_or_trades(name):
    data = markets()
    stop = 130 * 6 + 2
    prefix = {s: f.iloc[:stop] for s, f in data.items()}
    full, prices_f, quoted_f, _ = broad_targets(data)
    past, prices_p, quoted_p, _ = broad_targets(prefix)
    pd.testing.assert_frame_equal(past[name], full[name].iloc[:stop])
    pd.testing.assert_frame_equal(quoted_p, quoted_f.iloc[:stop])
    if name not in ("spot_control", "capped_control"):
        assert full[name].iloc[:89 * 6 + 5].eq(0).all().all()
    assert full[name].min().min() >= 0 and full[name].max().max() <= .25
    assert full[name].sum(axis=1).max() <= .5 + 1e-12
    eq_p, f_p, _ = run_intraday_spot(prices_p, past[name], quoted_mask=quoted_p, daily_control=True)
    eq_f, f_f, _ = run_intraday_spot(prices_f, full[name], quoted_mask=quoted_f, daily_control=True)
    pd.testing.assert_frame_equal(eq_p, eq_f.iloc[:stop])
    pd.testing.assert_frame_equal(f_p, f_f.loc[f_f.timestamp < quoted_f.index[stop]].reset_index(drop=True))
    assert reconcile_spot_path(prices_f, eq_f, f_f)["reconciled"]


def test_controls_ignore_other_assets_and_match_exact_previous_three_asset_targets():
    full, _, _, daily = broad_targets(markets())
    close = pd.DataFrame({s: daily[s].close for s in COHORT[:3]})
    for name, old in (("spot_control", "spot_control"), ("capped_control", "capped_allocation")):
        expected = MomentumRefinement(old).weights(close)
        expected.index += pd.Timedelta("20h")
        expected = expected.reindex(full[name].index, method="ffill", fill_value=0.)
        pd.testing.assert_frame_equal(full[name].loc[:, list(COHORT[:3])], expected)
        assert full[name].iloc[:, 3:].eq(0).all().all()


def test_one_missing_candle_invalidates_the_day_and_resets_ninety_day_eligibility():
    data = markets(200)
    data["SOLUSDT"].iloc[100 * 6 + 2] = np.nan
    daily = complete_daily(data)
    assert daily["SOLUSDT"].iloc[100].isna().all()
    close = pd.DataFrame({s: f.close for s, f in daily.items()})
    volume = pd.DataFrame({s: f.quote_volume for s, f in daily.items()})
    # Isolate from top-ten tie ranking to test the contiguous-history rule.
    active = liquidity_eligibility(close[["SOLUSDT"]], volume[["SOLUSDT"]])
    assert active.iloc[89, 0] and active.iloc[99, 0]
    assert not active.iloc[100:190, 0].any()
    assert active.iloc[190, 0]


def test_liquidity_threshold_and_top_ten_use_only_completed_historical_volume():
    dates = pd.date_range("2024-01-01", periods=110, freq="D", tz="UTC")
    close = pd.DataFrame(100., index=dates, columns=list(COHORT))
    volume = pd.DataFrame(12_000_000., index=dates, columns=close.columns)
    volume["BTCUSDT"] = 9_999_999.
    eligible = liquidity_eligibility(close, volume)
    assert eligible.iloc[:89].eq(False).all().all()
    assert eligible.iloc[89].sum() == 10 and not eligible.iloc[89, 0]
    assert eligible.iloc[89, 1:11].all() and not eligible.iloc[89, 11:].any()
    volume.iloc[100:, -1] = 1e12
    updated = liquidity_eligibility(close, volume)
    pd.testing.assert_frame_equal(updated.iloc[:100], eligible.iloc[:100])


def test_weekly_membership_waits_until_sunday_and_does_not_replace_a_daily_exit():
    dates = pd.date_range("2024-01-01", periods=15, freq="D", tz="UTC")
    score = pd.DataFrame([[4., 3., 2., 1.]] * len(dates), index=dates)
    active = score.gt(0)
    score.iloc[9:, 3] = 8.
    active.iloc[10:, 1] = False
    members = weekly_top(score, active, 3)
    assert not members.iloc[:6].any().any()
    assert members.iloc[7].tolist() == [True, True, True, False]
    assert members.iloc[9].tolist() == [True, True, True, False]
    assert members.iloc[10].tolist() == [True, False, True, False]
    assert members.iloc[14].tolist() == [True, False, True, True]


def test_polygon_notice_changes_only_completed_announcement_day_then_warmup_resets():
    data = markets(220, "2024-05-01")
    # Positive, liquid Polygon to make the event rule observable.
    data["POLYGON"]["close"] = np.linspace(100., 300., 220 * 6)
    data["POLYGON"]["quote_volume"] = 1e12
    before = data["POLYGON"].index < pd.Timestamp("2024-09-10", tz="UTC")
    after = data["POLYGON"].index >= pd.Timestamp("2024-09-14", tz="UTC")
    data["POLYGON"].loc[~before & ~after] = np.nan
    daily = complete_daily(data)
    close = pd.DataFrame({s: f.close for s, f in daily.items()})
    volume = pd.DataFrame({s: f.quote_volume for s, f in daily.items()})
    weights = BroadSpot("broad_inverse_vol").weights(close, volume)
    assert weights.loc["2024-08-27", "POLYGON"] > 0
    assert weights.loc["2024-08-28":, "POLYGON"].eq(0).all()
    prefix = BroadSpot("broad_inverse_vol").weights(close.loc[:"2024-08-27"], volume.loc[:"2024-08-27"])
    pd.testing.assert_frame_equal(prefix, weights.loc[:"2024-08-27"])


def test_volatility_scaling_uses_only_selected_covariance_never_increases_and_is_causal():
    rng = np.random.default_rng(922)
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(.002, .10, (100, 3)), axis=0)), index=dates)
    close[2] = np.nan
    weights = pd.DataFrame([[.25, .25, 0.]] * 100, index=dates)
    actual = reduce_volatility(weights, close)
    assert actual.iloc[:60].eq(0).all().all()
    assert 0 < actual.iloc[80].sum() < .5
    assert (actual <= weights).all().all()
    expected = reduce_volatility(weights.iloc[:81], close.iloc[:81])
    pd.testing.assert_frame_equal(expected, actual.iloc[:81])
    close.iloc[80, 0] = np.nan
    assert reduce_volatility(weights, close).iloc[80].eq(0).all()


def flat_market():
    dates = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
    frame = pd.DataFrame(100., index=dates, columns=["open", "high", "low", "close", "volume"])
    return {"BTCUSDT": frame}, pd.DataFrame(.25, index=dates, columns=["BTCUSDT"])


def test_quote_mask_all_true_keeps_default_prices_fills_and_every_metric():
    data, target = flat_market()
    mask = pd.DataFrame(True, index=target.index, columns=target.columns)
    a, f_a, s_a = run_intraday_spot(data, target)
    b, f_b, s_b = run_intraday_spot(data, target, quoted_mask=mask)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(f_a, f_b)
    assert s_a == s_b


def test_missing_market_keeps_owned_inventory_and_cash_but_cannot_execute_zero_price_fill():
    data, target = flat_market()
    mask = pd.DataFrame(True, index=target.index, columns=target.columns)
    mask.iloc[3:7] = False
    data["BTCUSDT"].iloc[3:7] = 0.
    target.iloc[2:] = 0.
    eq, fills, s = run_intraday_spot(data, target, quoted_mask=mask,
        fee_rate=0., slippage_bps=0., spread_bps=0., min_notional=0.)
    assert fills.timestamp.tolist() == [target.index[1], target.index[7]]
    assert fills.side.tolist() == ["buy", "sell"]
    assert eq.equity.iloc[3:7].eq(750.).all()
    assert eq.base_BTCUSDT.iloc[3:7].eq(2.5).all()
    assert s["held_unquoted_bars"] == s["held_unquoted_asset_bars"] == 4
    assert s["unquoted_asset_bars"] == 4
    assert s["max_intraday_drawdown_bound"] == pytest.approx(-.25)
    assert reconcile_spot_path(data, eq, fills)["reconciled"]


@pytest.mark.parametrize("kind", ["float", "nan", "nullable", "columns", "timestamps", "nonzero_absent", "zero_present"])
def test_invalid_quote_mask_or_sentinel_rejected(kind):
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
        mask.columns = ["OTHER"]
    elif kind == "timestamps":
        mask.index += pd.Timedelta("4h")
    elif kind == "nonzero_absent":
        mask.iloc[0, 0] = False
    else:
        data["BTCUSDT"].iloc[0] = 0.
    with pytest.raises(ValueError, match="mask|OHLCV"):
        run_intraday_spot(data, target, quoted_mask=mask)


def archive(dates, symbol, month, mutation=None):
    unit = "us" if month >= "2025-01" else "ms"
    opening = (dates.as_unit("ns").asi8 // (1_000 if unit == "us" else 1_000_000)).tolist()
    close = ((dates + pd.Timedelta("4h") - pd.Timedelta(1, unit=unit)).as_unit("ns").asi8
             // (1_000 if unit == "us" else 1_000_000)).tolist()
    raw = pd.DataFrame([[o, 100., 101., 99., 100., 10., c, 1_000., 2, 5., 500., 0] for o, c in zip(opening, close)])
    raw[8] = raw[8].astype(float)
    if mutation is not None:
        changed = mutation(raw)
        if changed is not None:
            raw = changed
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{symbol}-4h-{month}.csv", raw.to_csv(index=False, header=False))
    return payload.getvalue()


@pytest.mark.parametrize("month", ["2024-09", "2025-01"])
def test_partial_archive_keeps_real_absences_and_validates_timestamp_unit(month):
    dates = pd.date_range(f"{month}-01", periods=100, freq="4h", tz="UTC").as_unit("ns")
    dates = dates.delete(list(range(10, 50)))
    parsed = parse_partial_archive(archive(dates, "SOLUSDT", month), "SOLUSDT", month)
    assert pd.DatetimeIndex(parsed.timestamp).equals(dates)


@pytest.mark.parametrize("kind", ["duplicate", "unordered", "off_grid", "close_time", "ohlc", "taker", "trade_count"])
def test_bad_present_archive_bars_stop_instead_of_silently_dropping_assets(kind):
    dates = pd.date_range("2024-09-01", periods=3, freq="4h", tz="UTC")
    def mutate(raw):
        if kind == "duplicate":
            raw.iloc[1, 0] = raw.iloc[0, 0]
        elif kind == "unordered":
            raw.iloc[0, 0], raw.iloc[1, 0] = raw.iloc[1, 0], raw.iloc[0, 0]
        elif kind == "off_grid":
            raw.iloc[0, 0] += 1
        elif kind == "close_time":
            raw.iloc[0, 6] += 1
        elif kind == "ohlc":
            raw.iloc[0, 2] = 90.
        elif kind == "taker":
            raw.iloc[0, 10] = 2_000.
        else:
            raw.iloc[0, 8] = 1.5
    with pytest.raises(ValueError):
        parse_partial_archive(archive(dates, "MATICUSDT", "2024-09", mutate), "MATICUSDT", "2024-09")


def test_historical_aliases_exclude_new_terra_and_pre_listing_polygon_bar():
    results = {}
    for asset in ("TERRA_CLASSIC", "POLYGON"):
        for symbol, start, end in source_periods(asset):
            months = pd.date_range(start.normalize().replace(day=1), end - pd.Timedelta("1ns"), freq="MS")
            for month in months.strftime("%Y-%m"):
                first = pd.Timestamp(f"{month}-01", tz="UTC")
                dates = pd.date_range(first, first + pd.offsets.MonthBegin(1), freq="4h", inclusive="left")
                frame = pd.DataFrame({"timestamp": dates, **{c: 100. for c in COLUMNS}})
                results[(symbol, month)] = (frame, {"month": month})
        frame, _ = assemble_asset(asset, results)
        if asset == "TERRA_CLASSIC":
            assert frame.loc["2022-05-13 00:00", "market_symbol"] == "LUNAUSDT"
            assert frame.loc["2022-05-13 04:00":"2022-09-09 04:00", "close"].isna().all()
            assert frame.loc["2022-09-09 08:00", "market_symbol"] == "LUNCUSDT"
        else:
            assert frame.loc["2024-09-10 00:00", "market_symbol"] == "MATICUSDT"
            assert frame.loc["2024-09-10 04:00":"2024-09-13 08:00", "close"].isna().all()
            assert frame.loc["2024-09-13 12:00", "market_symbol"] == "POLUSDT"


def test_nonfixed_policy_is_rejected():
    with pytest.raises(ValueError):
        BroadSpot("post_result_variant")


@pytest.mark.parametrize("symbol,month,start,end", [("LUNAUSDT", "2022-05", "2022-05-13 00:00", "2022-05-13 00:40"),
    ("MATICUSDT", "2024-09", "2024-09-10 00:00", "2024-09-10 03:00")])
def test_only_exact_documented_terminal_partial_close_is_accepted(symbol, month, start, end):
    dates = pd.DatetimeIndex([pd.Timestamp(start, tz="UTC")])
    terminal = (pd.Timestamp(end, tz="UTC") - pd.Timedelta("1ms")).value // 1_000_000
    def valid(raw):
        raw.iloc[0, 6] = terminal
    assert len(parse_partial_archive(archive(dates, symbol, month, valid), symbol, month)) == 1
    def invalid(raw):
        raw.iloc[0, 6] = terminal - 1
    with pytest.raises(ValueError, match="close timestamp"):
        parse_partial_archive(archive(dates, symbol, month, invalid), symbol, month)


@pytest.mark.parametrize("problem", [None, "daily_conflict", "monthly_conflict", "bad_daily_checksum"])
def test_identical_duplicate_requires_checksum_verified_agreement_of_entire_separate_daily_archive(monkeypatch, tmp_path, problem):
    dates = pd.date_range("2025-01-06", periods=6, freq="4h", tz="UTC")
    def repeated(raw):
        duplicate = raw.iloc[:1].copy()
        if problem == "monthly_conflict":
            duplicate.iloc[0, 4] = 100.5
        return pd.concat([duplicate, raw], ignore_index=True)
    def daily_change(raw):
        if problem == "daily_conflict":
            raw.iloc[-1, 4] = 100.5
    monthly = archive(dates, "AVAXUSDT", "2025-01", repeated)
    daily = archive(dates, "AVAXUSDT", "2025-01-06", daily_change)
    payloads = {}
    for scope, period, data in (("monthly", "2025-01", monthly), ("daily", "2025-01-06", daily)):
        stem = f"AVAXUSDT-4h-{period}"
        url = f"https://data.binance.vision/data/spot/{scope}/klines/AVAXUSDT/4h/{stem}.zip"
        payloads[url] = data
        checksum = "0" * 64 if scope == "daily" and problem == "bad_daily_checksum" else hashlib.sha256(data).hexdigest()
        payloads[url + ".CHECKSUM"] = f"{checksum}  {stem}.zip\n".encode()
    monkeypatch.setattr("scripts.download_spot_universe._get", lambda url, cache: payloads[url])
    if problem:
        with pytest.raises(ValueError, match="Conflicting|corroborate|SHA-256"):
            fetch_month("AVAXUSDT", "2025-01", tmp_path)
    else:
        parsed, proof = fetch_month("AVAXUSDT", "2025-01", tmp_path)
        assert len(parsed) == 6 and proof["published_rows"] == 7
        assert proof["duplicate_corroboration"][0]["monthly_duplicate_rows_removed"] == 1
        assert proof["duplicate_corroboration"][0]["sha256"] == hashlib.sha256(daily).hexdigest()
    with pytest.raises(ValueError, match="Unverified duplicate"):
        parse_partial_archive(monthly, "AVAXUSDT", "2025-01")


def test_freeze_rejects_higher_profit_with_unquoted_inventory_or_drawdown():
    def summary(net, dd, missing=0):
        return {"net_pnl": net, "max_intraday_drawdown_bound": dd, "held_unquoted_asset_bars": missing}
    dev = {"good": summary(200., -.19), "gap": summary(900., -.10, 1),
           "risk": summary(1_000., -.201), "loss": summary(-20., -.05)}
    assert freeze_broad(dev) == "good"
    assert freeze_broad({"loss": dev["loss"], "gap": dev["gap"]}) == "cash"


def test_block_interval_pairs_aligned_returns_and_is_deterministic():
    dates = pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC")
    control = pd.Series(1000., index=dates)
    candidate = pd.Series(1000. * np.exp(.001 * np.arange(1, 101)), index=dates)
    interval = paired_block_interval(candidate, control)
    assert interval == paired_block_interval(candidate, control)
    assert interval["lower"] == pytest.approx(.365)
    assert interval["upper"] == pytest.approx(.365)
    assert interval["positive_lower_bound"]
    zero = paired_block_interval(control, control)
    assert zero["lower"] == zero["upper"] == 0.
    assert not zero["positive_lower_bound"]
    with pytest.raises(ValueError, match="Aligned"):
        paired_block_interval(candidate.iloc[1:], control)
