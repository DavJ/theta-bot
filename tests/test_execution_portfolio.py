"""Economic accounting, execution clocks and portfolio causality invariants."""
import numpy as np
import pandas as pd
import pytest

from spot_bot.backtest.portfolio import run_portfolio_backtest
from spot_bot.core.cost_model import compute_cost_per_turnover, compute_round_trip_cost
from spot_bot.core.engine import EngineParams, run_step_simulated
from spot_bot.core.hysteresis import compute_hysteresis_threshold
from spot_bot.core.types import MarketBar, PortfolioState
from spot_bot.portfolio.trend import PORTFOLIO_APPROACHES, TrendPortfolio
from spot_bot.strategies.base import Intent
from spot_bot.strategies.multi_strategy import MultiStrategy


def markets(n=500):
    dates = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    rng = np.random.default_rng(75)
    result = {}
    for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT"):
        close = 100 * np.exp(np.cumsum(rng.normal(.0005, .025, n)))
        open_ = np.r_[100, close[:-1]]
        result[symbol] = pd.DataFrame({"open": open_, "close": close,
                                       "high": np.maximum(open_, close) * 1.01,
                                       "low": np.minimum(open_, close) * .99,
                                       "volume": 100}, index=dates)
    return result


def test_one_way_cost_matches_market_fill_and_round_trip_counts_both_legs():
    class Long:
        def generate_intent(self, _):
            return Intent(.3, reason="test", diagnostics={})
    state = PortfolioState(1000, 0, 1000, 0)
    params = EngineParams(fee_rate=.001, slippage_bps=5, spread_bps=2, execution_policy="market")
    # Large future range/close cannot change a known-open market fill.
    bar = MarketBar(1, 100, 1000, 10, 500, 100)
    plan, fill, _, _ = run_step_simulated(bar, pd.DataFrame(), state, Long(), params, .01, .01)
    assert plan.order_type == "market" and plan.limit_price is None
    assert fill.avg_price == pytest.approx(100.06)
    observed = (fill.fee_paid + fill.slippage_paid) / (abs(fill.filled_base) * bar.open)
    # The fee is charged on impacted fill notional, adding a second-order term.
    assert observed == pytest.approx(compute_cost_per_turnover(.001, 5, 2), abs=1e-6)
    assert compute_round_trip_cost(.001, 5, 2) == pytest.approx(.0032)


@pytest.mark.parametrize("ratio,floor,cap", [(0.01,.1,.3),(100,.02,.3),(1,.3,.3)])
def test_documented_hysteresis_bounds_are_enforced(ratio, floor, cap):
    value = compute_hysteresis_threshold(rv_current=ratio, rv_ref=1, fee_rate=.001,
                                         slippage_bps=5, spread_bps=2, hyst_k=5,
                                         hyst_floor=floor, max_delta_e_min=cap)
    assert floor <= value <= cap


def test_completed_daily_signal_is_available_at_final_candle_close():
    frame = pd.DataFrame({"close": 100.0}, index=pd.date_range("2024-01-01", periods=24*70,
                                                              freq="1h", tz="UTC"))
    frame.iloc[-1, 0] = 200
    result = MultiStrategy("breakout").generate_series(frame)
    assert result.iloc[:-1].eq(0).all()
    assert result.iloc[-1] == .3
    # At 23:00's close, midnight's daily signal is known; it can execute on the
    # following midnight open, not an extra hour later.
    extended = pd.concat([frame, pd.DataFrame({"close": [200.]},
                          index=[frame.index[-1] + pd.Timedelta("1h")])])
    assert MultiStrategy("breakout").generate_series(extended).shift(1).iloc[-1] == .3


@pytest.mark.parametrize("name", PORTFOLIO_APPROACHES)
def test_portfolio_future_prices_do_not_change_past_targets_or_replay(name):
    data = markets()
    close = pd.DataFrame({s: f.close for s, f in data.items()})
    policy = TrendPortfolio(name)
    pd.testing.assert_frame_equal(policy.weights(close.iloc[:430]), policy.weights(close).iloc[:430])
    targets = policy.weights(close)
    assert targets.sum(axis=1).max() <= .3 + 1e-12
    cap = .3 if name == "btc_breakout" else .15
    assert targets.max().max() <= cap + 1e-12
    a, t_a, _ = run_portfolio_backtest({s: f.iloc[:430] for s, f in data.items()}, policy)
    b, t_b, _ = run_portfolio_backtest(data, policy)
    pd.testing.assert_frame_equal(a, b.iloc[:430])
    pd.testing.assert_frame_equal(t_a, t_b.loc[t_b.timestamp < data["BTCUSDT"].index[430]].reset_index(drop=True))
    assert b.usdt.min() >= -1e-8


def test_shared_cash_account_charges_each_fill_once(monkeypatch):
    data = markets(17)
    for frame in data.values():
        frame[["open", "high", "low", "close"]] = 100.
        frame.index = pd.date_range("2024-01-01", periods=17, freq="D", tz="UTC")
    monkeypatch.setattr(TrendPortfolio, "weights", lambda self, close: close * 0 + .1)
    equity, trades, summary = run_portfolio_backtest(data, TrendPortfolio("horizons_equal"))
    assert len(trades) == 3
    assert trades.timestamp.eq(pd.Timestamp("2024-01-08", tz="UTC")).all()
    assert summary["net_pnl"] == pytest.approx(-summary["fees_paid_total"] - summary["slippage_paid_total"])
    assert equity.usdt.iloc[-1] + 100 * sum(equity[f"base_{s}"].iloc[-1] for s in data) == pytest.approx(summary["final_equity"])


def test_inactive_asset_exits_on_next_open_without_waiting_for_weekly_rebalance(monkeypatch):
    data = markets(17)
    dates = pd.date_range("2024-01-01", periods=17, freq="D", tz="UTC")
    for frame in data.values():
        frame.index = dates
        frame[["open", "high", "low", "close"]] = 100.
        frame.loc[dates[9]:, ["open", "high", "low", "close"]] = 90.
    def weights(self, close):
        result = close * 0 + .1
        result.iloc[8:] = 0.
        return result
    monkeypatch.setattr(TrendPortfolio, "weights", weights)
    equity, trades, _ = run_portfolio_backtest(data, TrendPortfolio("horizons_equal"))
    sells = trades.loc[trades.side == "sell"]
    assert len(sells) == 3
    assert sells.timestamp.eq(dates[9]).all() and dates[9].dayofweek != 0
    assert equity.filter(like="base_").iloc[-1].eq(0).all()


def test_missing_asset_day_is_rejected():
    data = markets(100)
    data["ETHUSDT"] = data["ETHUSDT"].drop(data["ETHUSDT"].index[50])
    with pytest.raises(ValueError, match="identical dates"):
        run_portfolio_backtest(data, TrendPortfolio("breakout_equal"))
