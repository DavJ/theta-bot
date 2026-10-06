"""Hand-computed derivatives ledger, information clock and source regressions."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.download_futures_research import fetch, parse_futures_archive
from spot_bot.backtest.perpetual import apply_perpetual_fill, funding_payment_bound, run_perpetual_backtest
from spot_bot.portfolio.signed import SIGNED_APPROACHES, SignedPortfolio


@pytest.mark.parametrize("q,entry,delta,fill,expected", [
    (0., 0., 2., 100., (2., 100., 0.)),
    (0., 0., -2., 100., (-2., 100., 0.)),
    (2., 100., 1., 130., (3., 110., 0.)),
    (-2., 100., -1., 70., (-3., 90., 0.)),
    (2., 100., -1., 120., (1., 100., 20.)),
    (-2., 100., 1., 80., (-1., 100., 20.)),
    (2., 100., -2., 80., (0., 0., -40.)),
    (-2., 100., 2., 120., (0., 0., -40.)),
    (2., 100., -3., 120., (-1., 120., 40.)),
    (-2., 100., 3., 80., (1., 80., 40.)),
])
def test_signed_entry_realizes_only_closed_inventory(q, entry, delta, fill, expected):
    np.testing.assert_allclose(apply_perpetual_fill(q, entry, delta, fill), expected)


@pytest.mark.parametrize("before,after,rate,expected", [
    (2., 2., .01, 2.4), (-2., -2., .01, -1.6),
    (2., 2., -.01, -1.6), (-2., -2., -.01, 2.4),
    (0., -2., .01, 0.), (2., 0., .01, 2.4),
    (-2., 2., .01, 2.4),
])
def test_funding_signs_and_uncertain_boundary_inventory(before, after, rate, expected):
    assert funding_payment_bound(before, after, rate, 120., 80.) == pytest.approx(expected)


def account(days=4, symbols=("BTCUSDT",)):
    daily = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")
    hourly = pd.date_range(daily[0], periods=24 * days, freq="h")
    execution = {s: pd.DataFrame(100., index=daily, columns=["open", "high", "low", "close"]) for s in symbols}
    marks = {s: pd.DataFrame(100., index=hourly, columns=["open", "high", "low", "close"]) for s in symbols}
    funding = {s: pd.DataFrame(columns=["event_time", "last_funding_rate"]) for s in symbols}
    targets = pd.DataFrame(.5, index=daily, columns=symbols)
    close = pd.DataFrame(100., index=daily, columns=symbols)
    return execution, marks, funding, targets, close


def replay(parts, **kwargs):
    return run_perpetual_backtest(*parts, fee_rate=0., slippage_bps=0., spread_bps=0.,
        min_notional=0., rebalance_band=0., volatility_target=None, **kwargs)


@pytest.mark.parametrize("direction", [1., -1.])
def test_mark_to_market_and_full_close_with_daily_target_delay(direction):
    parts = account(3)
    execution, marks, funding, targets, _ = parts
    targets.iloc[:, :] = 0.
    targets.iloc[0, 0] = .5 * direction
    marks["BTCUSDT"].loc[marks["BTCUSDT"].index >= targets.index[1], "close"] = 120.
    marks["BTCUSDT"].loc[marks["BTCUSDT"].index >= targets.index[1], "high"] = 120.
    marks["BTCUSDT"].loc[marks["BTCUSDT"].index >= targets.index[2], ["open", "low"]] = 120.
    execution["BTCUSDT"].iloc[2, :] = 120.
    equity, trades, _, summary = replay(parts)
    assert trades.timestamp.tolist() == [targets.index[1], targets.index[2]]
    assert trades.signed_delta.tolist() == [5 * direction, -5 * direction]
    assert equity.equity.tolist() == pytest.approx([1000., 1000 + 100 * direction, 1000 + 100 * direction])
    assert equity.collateral_cash.iloc[1] == 1000.
    assert equity.unrealized_pnl.iloc[1] == pytest.approx(100 * direction)
    assert equity.realized_pnl_cumulative.iloc[-1] == pytest.approx(100 * direction)
    assert summary["total_return"] == pytest.approx(.1 * direction)


def test_fill_impact_fee_and_funding_debit_each_exactly_once():
    parts = account(3)
    parts[3].iloc[:, :] = 0.
    parts[3].iloc[0, 0] = .5
    ts = parts[3].index[1] + pd.Timedelta("8h") + pd.Timedelta("9ms")
    parts[2]["BTCUSDT"] = pd.DataFrame({"event_time": [ts], "last_funding_rate": [.001]})
    equity, trades, settled, summary = run_perpetual_backtest(*parts, fee_rate=.01,
        slippage_bps=10., spread_bps=0., min_notional=0., rebalance_band=0., volatility_target=None)
    # +5 at 100.1, -5 at 99.9; impact 1, fees 10, funding .5.
    assert equity.equity.iloc[-1] == pytest.approx(988.5)
    assert summary["fees_paid_total"] == pytest.approx(10.)
    assert summary["slippage_paid_total"] == pytest.approx(1.)
    assert summary["funding_net_paid_total"] == pytest.approx(.5)
    assert settled.timestamp.iloc[0] == ts
    assert trades.realized_pnl.sum() == pytest.approx(-1.)
    assert equity.collateral_cash.iloc[-1] == pytest.approx(988.5)


def test_same_open_funding_and_mark_extremes_do_not_resize_orders():
    base, changed = account(3), account(3)
    ts = changed[3].index[1]
    changed[2]["BTCUSDT"] = pd.DataFrame({"event_time": [ts + pd.Timedelta("6ms")], "last_funding_rate": [.05]})
    changed[1]["BTCUSDT"].loc[ts, ["high", "low", "close"]] = [150., 50., 110.]
    e1, t1, _, _ = replay(base)
    e2, t2, _, _ = replay(changed)
    first = ["signed_delta", "decision_nav", "price", "fee"]
    pd.testing.assert_series_equal(t1.iloc[0][first], t2.iloc[0][first])
    assert e2.equity.iloc[1] < e1.equity.iloc[1]
    assert t2.iloc[1].decision_nav < t1.iloc[0].decision_nav


def test_short_hourly_bounds_use_high_as_loss_and_low_as_gain():
    parts = account(2)
    parts[3].iloc[:, :] = -.5
    ts = parts[3].index[1] + pd.Timedelta("12h")
    parts[1]["BTCUSDT"].loc[ts, ["high", "low"]] = [120., 80.]
    equity, _, _, summary = replay(parts)
    assert equity.hourly_low_nav_bound.iloc[-1] == pytest.approx(900.)
    assert equity.hourly_high_nav_bound.iloc[-1] == pytest.approx(1100.)
    assert summary["max_intraday_drawdown_bound"] == pytest.approx(900 / 1100 - 1)
    assert summary["maxDD"] == 0.


def test_offsetting_funding_has_intermediate_cash_bounds():
    parts = account(2, ("BTCUSDT", "ETHUSDT"))
    parts[3].iloc[:, :] = [.25, -.25]
    ts = parts[3].index[1] + pd.Timedelta("8h")
    for s in parts[2]:
        parts[2][s] = pd.DataFrame({"event_time": [ts + pd.Timedelta("9ms")], "last_funding_rate": [.01]})
    equity, _, settled, _ = replay(parts)
    assert settled.payment.sum() == pytest.approx(0.)
    assert equity.equity.iloc[-1] == pytest.approx(1000.)
    assert equity.hourly_low_nav_bound.iloc[-1] == pytest.approx(997.5)
    assert equity.hourly_high_nav_bound.iloc[-1] == pytest.approx(1002.5)


def test_engine_prefix_unchanged_by_future_marks_targets_or_funding():
    parts = account(4)
    short = tuple({s: f.iloc[:48].copy() for s, f in parts[1].items()} if i == 1
                  else {s: f.iloc[:2].copy() for s, f in p.items()} if i == 0
                  else p.copy() if i == 2 else p.iloc[:2].copy() for i, p in enumerate(parts))
    full_e, full_t, _, _ = replay(parts)
    e, t, _, _ = replay(short)
    pd.testing.assert_frame_equal(full_e.iloc[:2].reset_index(drop=True), e)
    pd.testing.assert_frame_equal(full_t.loc[full_t.timestamp < parts[3].index[2]].reset_index(drop=True), t)


def test_prior_covariance_and_completed_target_are_only_risk_inputs():
    a, b = account(70), account(70)
    prices = 100 * np.exp(np.sin(np.arange(70)) * .02)
    a[4].iloc[:, 0] = prices
    b[4].iloc[:, 0] = prices
    b[4].iloc[61:, 0] *= 8.
    b[3].iloc[61:, 0] = -.5
    options = dict(fee_rate=0., slippage_bps=0., spread_bps=0., min_notional=0., rebalance_band=0.)
    _, t1, _, _ = run_perpetual_backtest(*a, **options)
    _, t2, _, _ = run_perpetual_backtest(*b, **options)
    end = a[3].index[62]
    pd.testing.assert_frame_equal(t1.loc[t1.timestamp < end].reset_index(drop=True),
                                  t2.loc[t2.timestamp < end].reset_index(drop=True))


def test_missing_hourly_mark_is_rejected_instead_of_filled():
    parts = account(3)
    parts[1]["BTCUSDT"] = parts[1]["BTCUSDT"].drop(parts[1]["BTCUSDT"].index[8])
    with pytest.raises(ValueError, match="grid"):
        replay(parts)


def test_pandas_mixed_timestamp_units_share_one_decision_grid():
    parts = account(2)
    for f in [*parts[0].values(), *parts[1].values(), parts[3], parts[4]]:
        f.index = f.index.astype("datetime64[us, UTC]")
    equity, trades, _, _ = replay(parts)
    assert len(equity) == 2 and len(trades) == 1


def close_history(n=220):
    rng = np.random.default_rng(721)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .02, (n, 3)), axis=0)),
        index=pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC"),
        columns=["BTCUSDT", "ETHUSDT", "BNBUSDT"])


@pytest.mark.parametrize("name", (*SIGNED_APPROACHES, "perp_long_control"))
def test_completed_signed_signals_are_prefix_invariant_and_capped(name):
    close = close_history()
    policy = SignedPortfolio(name)
    full, prefix = policy.weights(close), policy.weights(close.iloc[:160])
    pd.testing.assert_frame_equal(full.iloc[:160], prefix)
    assert np.isfinite(full.to_numpy()).all()
    assert full.abs().sum(axis=1).max() <= 1. + 1e-12
    assert full.abs().max().max() <= .5 + 1e-12
    assert full.iloc[:60].eq(0.).all().all()


def test_relative_is_dollar_neutral_and_flat_when_rank_spread_vanishes():
    close = close_history()
    weights = SignedPortfolio("relative").weights(close)
    np.testing.assert_allclose(weights.sum(axis=1), 0.)
    assert weights.iloc[100].abs().sum() == 1.
    identical = pd.concat([close.BTCUSDT] * 3, axis=1)
    identical.columns = close.columns
    assert SignedPortfolio("relative").weights(identical).eq(0.).all().all()


def test_pair_shock_uses_preceding_regression_not_its_own_outlier():
    n = np.arange(91)
    x = 5. + n * .004
    y = 1.5 * x + .01 * np.sin(n * 1.4)
    beta, alpha = np.linalg.lstsq(np.column_stack([x[:90], np.ones(90)]), y[:90], rcond=None)[0]
    sigma = np.std(y[:90] - alpha - beta * x[:90], ddof=0)
    y[90] = alpha + beta * x[90] + 2.5 * sigma
    close = pd.DataFrame({"BTCUSDT": np.exp(x), "ETHUSDT": np.exp(y), "BNBUSDT": np.exp(x + .3)},
        index=pd.date_range("2025-01-01", periods=91, freq="D", tz="UTC"))
    w = SignedPortfolio("pairs").weights(close).iloc[-1]
    np.testing.assert_allclose(w, [.5 * beta / (1 + beta), -.5 / (1 + beta), 0.], atol=1e-8)
    assert SignedPortfolio("pairs").weights(close).iloc[:90].eq(0.).all().all()


def test_hourly_margin_bound_flags_unobserved_short_squeeze():
    parts = account(2)
    parts[3].iloc[:, :] = -.5
    ts = parts[3].index[1] + pd.Timedelta("12h")
    parts[1]["BTCUSDT"].loc[ts, "high"] = 400.
    equity, _, _, summary = replay(parts)
    assert summary["margin_breach_hours"] == 1
    assert equity.equity.iloc[-1] == 1000.
    assert summary["max_intraday_drawdown_bound"] < -1.
    parts[1]["BTCUSDT"].loc[ts, "close"] = 400.
    with pytest.raises(ValueError, match="NAV nonpositive"):
        replay(parts)


def archive_bytes(symbol="BTCUSDT", day="2025-01-01", missing=(), duplicate=False, header=True):
    dates = pd.date_range(day, periods=24, freq="h", tz="UTC")
    rows = [[int(t.timestamp() * 1000), 100., 101., 99., 100., 0., 0, 0, 0, 0, 0, 0]
            for i, t in enumerate(dates) if i not in missing]
    if duplicate:
        rows.append(rows[-1])
    body = pd.DataFrame(rows).to_csv(index=False, header=False)
    if header:
        body = "open_time,open,high,low,close,volume,close_time,qv,count,tbv,tqv,ignore\n" + body
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(f"{symbol}-1h-{day}.csv", body)
    return out.getvalue()


@pytest.mark.parametrize("header", [False, True])
def test_post_2025_futures_archive_keeps_millisecond_time(header):
    frame = parse_futures_archive(archive_bytes(header=header), "BTCUSDT", "1h", "2025-01-01")
    assert len(frame) == 24
    assert frame.timestamp.iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")


@pytest.mark.parametrize("kwargs", [{"missing": (8,)}, {"duplicate": True}])
def test_archive_grid_gaps_and_duplicate_bars_fail(kwargs):
    with pytest.raises(ValueError, match="Missing, duplicate"):
        parse_futures_archive(archive_bytes(**kwargs), "BTCUSDT", "1h", "2025-01-01")


def test_futures_daily_repair_must_have_matching_official_checksum(monkeypatch, tmp_path):
    # A monthly payload with one absent day; repair must be official and complete.
    dates = pd.date_range("2025-02-01", "2025-03-01", freq="h", inclusive="left", tz="UTC")
    rows = [[int(t.timestamp() * 1000), 100., 101., 99., 100., 0., 0, 0, 0, 0, 0, 0]
            for t in dates if t.day != 2]
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("BTCUSDT-1h-2025-02.csv", pd.DataFrame(rows).to_csv(index=False, header=False))
    monthly, daily = out.getvalue(), archive_bytes(day="2025-02-02")
    def source(url, cache):
        blob = daily if "/daily/" in url else monthly
        return hashlib.sha256(blob).hexdigest().encode() if url.endswith(".CHECKSUM") else blob
    monkeypatch.setattr("scripts.download_futures_research._get", source)
    frame, meta = fetch(("mark", "BTCUSDT", "2025-02"), tmp_path)
    assert len(frame) == 672 and meta["missing_monthly_bars"] == 24
    assert len(meta["official_daily_repairs"]) == 1
    monkeypatch.setattr("scripts.download_futures_research._get", lambda url, _: b"0" * 64 if url.endswith(".CHECKSUM") else monthly)
    with pytest.raises(ValueError, match="checksum"):
        fetch(("mark", "BTCUSDT", "2025-02"), tmp_path)


def test_selection_is_frozen_before_later_results_and_does_not_reselect(monkeypatch, tmp_path):
    from scripts import research_long_short as study
    close = close_history(110)
    markets = {s: pd.DataFrame({"close": close[s]}) for s in close}
    scores = {"spot_control": 5., "perp_long_control": 10., "signed_m30": 99.,
              "signed_horizons": 15., "ema": 17., "breakout": 0., "relative": -1., "pairs": 12., "blend": 8.}
    calls = []
    def fake_replay(markets, futures, close, targets, name, out, label, start, end, costs=study.COSTS):
        calls.append((name, label))
        development = label == "development"
        if not development:
            saved = json.loads((out / "frozen_long_short_selection.json").read_text())
            assert saved["frozen_candidate"] == "signed_horizons"
        pnl = scores[name] if development else (100. if name == "relative" else 50. if name == "spot_control" else 1.)
        return {"net_pnl": pnl, "drawdown_within_20pct_bound": name != "signed_m30",
                "collateral_screen_passed": name != "ema"}
    monkeypatch.setattr(study, "_replay", fake_replay)
    report = study.research(markets, {}, tmp_path, jobs=1)
    assert report["frozen_candidate"] == "signed_horizons"
    assert len(calls) == 81
    assert not report["gates"]["improves_continuous_later"]
    assert not report["historical_screen_passed"]
    assert report["scenarios"]["relative"]["periods"]["continuous"]["net_pnl"] == 100.


def test_real_research_path_reconciles_and_serializes_selection_metrics(tmp_path):
    from scripts.research_long_short import _replay
    parts = account(7, ("BTCUSDT", "ETHUSDT", "BNBUSDT"))
    parts[3].iloc[:, :] = .1
    futures = dict(zip(("trade", "mark", "funding"), parts[:3]))
    summary = _replay({}, futures, parts[4], {"perp_long_control": parts[3]}, "perp_long_control", tmp_path,
        "development", parts[3].index[0], parts[3].index[-1] + pd.Timedelta("1D"))
    restored = json.loads(json.dumps(summary, allow_nan=False))
    assert restored["reconciled"] and restored["collateral_screen_passed"]
    assert restored["trades_count"] == 3
