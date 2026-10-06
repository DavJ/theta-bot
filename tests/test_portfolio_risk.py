"""Risk timing, correlated exposure, actual loss fills and unreplenished budgets."""
import numpy as np
import pandas as pd
import pytest

from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.portfolio.risk import PortfolioRisk
from spot_bot.portfolio.trend import TrendPortfolio
from tests.test_execution_portfolio import markets


def flat_markets(n=30):
    data = markets(n)
    for frame in data.values():
        frame.index = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        frame[["open", "high", "low", "close"]] = 100.
    return data


ZERO_COSTS = dict(fee_rate=0, slippage_bps=0, spread_bps=0)


def test_covariance_counts_correlated_assets_and_is_prefix_causal():
    dates = pd.date_range("2024-01-01", periods=12, tz="UTC")
    prices = 100 * np.cumprod(1 + np.array([0, .04, -.02, .01, -.03, .04,
                                          -.01, .02, -.02, .03, .02, -.01]))
    close = pd.DataFrame({"BTCUSDT": prices, "ETHUSDT": prices}, index=dates)
    risk = PortfolioRisk(volatility_target=.1, volatility_window=5, covariance_shrinkage=0)
    cov = risk.covariances(close)
    np.testing.assert_allclose(risk.covariances(close.iloc[:10]), cov[:10], equal_nan=True)
    weights = pd.Series([.5, .5], index=close.columns)
    final, info = risk.constrain(weights, cov[-1], 1000, 1000)
    expected = close.BTCUSDT.pct_change().iloc[-5:].std(ddof=0) * np.sqrt(365)
    assert info["estimated_annual_volatility"] == pytest.approx(expected)
    assert final.sum() == pytest.approx(.1 / expected)
    warm, _ = risk.constrain(weights, cov[2], 1000, 1000)
    assert warm.eq(0).all()


@pytest.mark.parametrize("risk", [PortfolioRisk(volatility_target=.25),
    PortfolioRisk(volatility_target=.25, drawdown_limit=.2, cushion_multiplier=5)])
def test_risk_replay_future_data_cannot_change_past_fills_or_account(risk):
    data = markets(350)
    policy = TrendPortfolio("momentum_vol", max_exposure=1, asset_cap=.5)
    prefix, prefix_trades, _ = run_portfolio_backtest({s: f.iloc[:300] for s, f in data.items()}, policy, risk=risk)
    full, full_trades, _ = run_portfolio_backtest(data, policy, risk=risk)
    pd.testing.assert_frame_equal(prefix, full.iloc[:300])
    pd.testing.assert_frame_equal(prefix_trades, full_trades.loc[
        full_trades.timestamp < data["BTCUSDT"].index[300]].reset_index(drop=True))
    assert full.usdt.min() >= -1e-8


def test_current_day_range_is_only_available_for_next_day_risk_orders(monkeypatch):
    data = flat_markets()
    monkeypatch.setattr(TrendPortfolio, "weights", lambda self, close: close * 0 + .3)
    policy = TrendPortfolio("momentum_vol", max_exposure=1, asset_cap=.5, rebalance_band=.2)
    risk = PortfolioRisk(drawdown_limit=.2, cushion_multiplier=5)
    baseline, base_trades, _ = run_portfolio_backtest(data, policy, risk=risk, **ZERO_COSTS)
    event = next(iter(data.values())).index[8]  # Tuesday after first Monday allocation.
    for frame in data.values():
        frame.loc[event, "high"] = 112.
    equity, trades, _ = run_portfolio_backtest(data, policy, risk=risk, **ZERO_COSTS)
    pd.testing.assert_frame_equal(base_trades.loc[base_trades.timestamp <= event].reset_index(drop=True),
                                  trades.loc[trades.timestamp <= event].reset_index(drop=True))
    reductions = trades.loc[trades.risk_reduction]
    assert len(reductions) == 3 and reductions.timestamp.eq(event + pd.Timedelta("1D")).all()
    assert (reductions.notional < .2 * reductions.decision_nav).all()
    assert reductions.timestamp.iloc[0].dayofweek != policy.rebalance_weekday
    assert equity.loc[9, "exposure"] < baseline.loc[9, "exposure"]


def test_gap_breaches_budget_then_sells_at_actual_open_without_reset(monkeypatch):
    data = flat_markets()
    monkeypatch.setattr(TrendPortfolio, "weights", lambda self, close: close * 0 + .3)
    for frame in data.values():
        frame.iloc[8:, frame.columns.get_indexer(["open", "high", "low", "close"])] = 50.
    equity, trades, summary = run_portfolio_backtest(
        data, TrendPortfolio("momentum_vol", max_exposure=1, asset_cap=.5),
        risk=PortfolioRisk(drawdown_limit=.2), **ZERO_COSTS)
    sells = trades.loc[trades.side == "sell"]
    assert len(sells) == 3 and sells.price.eq(50).all()
    np.testing.assert_allclose(sells.decision_nav, 550, atol=1e-10)
    assert summary["maxDD"] == pytest.approx(-.45)
    assert equity.equity.iloc[-1] == pytest.approx(550)
    np.testing.assert_allclose(equity.risk_floor.iloc[8:], 820, atol=1e-10)
    assert equity.decision_peak_bound.iloc[8:].eq(1000).all()
    assert trades.loc[trades.timestamp > sells.timestamp.iloc[0]].empty


def test_intraday_bound_detects_loss_hidden_by_unchanged_closes(monkeypatch):
    data = flat_markets()
    monkeypatch.setattr(TrendPortfolio, "weights", lambda self, close: close * 0 + .3)
    for frame in data.values():
        frame.iloc[8, frame.columns.get_indexer(["high", "low"])] = [150., 60.]
    _, _, summary = run_portfolio_backtest(
        data, TrendPortfolio("momentum_vol", max_exposure=1, asset_cap=.5), **ZERO_COSTS)
    assert summary["maxDD"] == 0
    assert summary["max_observed_drawdown"] == 0
    assert summary["max_intraday_drawdown_bound"] == pytest.approx(640 / 1450 - 1)


def test_minimum_notional_can_leave_exposure_above_requested_budget(monkeypatch):
    data = flat_markets()
    monkeypatch.setattr(TrendPortfolio, "weights", lambda self, close: close * 0 + .3)
    for frame in data.values():
        frame.iloc[8, frame.columns.get_loc("high")] = 100.1
    equity, trades, _ = run_portfolio_backtest(
        data, TrendPortfolio("momentum_vol", max_exposure=1, asset_cap=.5),
        risk=PortfolioRisk(drawdown_limit=.2), **ZERO_COSTS)
    assert len(trades) == 3
    assert equity.loc[9, "exposure"] == pytest.approx(.9)
    assert equity.loc[9, "target_exposure"] < equity.loc[9, "exposure"]
    assert equity.loc[9, "exposure"] > equity.loc[9, "risk_budget_exposure"]


@pytest.mark.parametrize("kwargs", [{"drawdown_limit": .01}, {"volatility_target": 0},
    {"volatility_window": 1}, {"covariance_shrinkage": 2}, {"cushion_multiplier": np.nan}])
def test_invalid_risk_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PortfolioRisk(**kwargs)
