"""Daily aggregation, shared intent paths, netting and explicit risk exits."""
import numpy as np
import pandas as pd
import pytest

from spot_bot.core import EngineParams, MarketBar, PortfolioState, run_step_simulated
from spot_bot.core.legacy_adapter import LegacyStrategyAdapter
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.strategies.base import Intent
from spot_bot.strategies.multi_strategy import MULTI_APPROACHES, MultiStrategy
from spot_bot.backtest.fast_backtest import run_backtest
from spot_bot.run_live import build_parser, compute_step
from spot_bot.features import FeatureConfig


def features(n=24 * 240):
    rng = np.random.default_rng(8)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, n)))
    return pd.DataFrame({"close": close, "open": close, "high": close * 1.005,
                         "low": close * 0.995, "volume": 10,
                         "C": 0.6, "S": 0.5, "rv": 0.004, "psi": 0.2},
                        index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"))


@pytest.mark.parametrize("approach", MULTI_APPROACHES)
def test_future_days_do_not_change_prefix_and_single_intent_matches_series(approach):
    frame = features()
    prefix = frame.iloc[:24 * 220 + 12]
    strategy = MultiStrategy(approach)
    a = strategy.generate_series(prefix)
    b = strategy.generate_series(frame)
    pd.testing.assert_series_equal(a, b.iloc[:len(prefix)])
    assert a.between(0, 0.3).all()
    assert a.iloc[-1] == pytest.approx(strategy.generate_intent(prefix).desired_exposure)


def test_unfinished_daily_close_is_not_broadcast_backwards():
    frame = features(n=24 * 100)
    changed = frame.copy()
    day = changed.index[-1].normalize()
    changed.loc[day + pd.Timedelta(hours=12):, "close"] *= 2
    strategy = MultiStrategy("ema_trend")
    pd.testing.assert_series_equal(strategy.generate_series(frame), strategy.generate_series(changed))


def test_insufficient_fusion_history_stays_cash_and_parser_exposes_new_choices():
    frame = features(n=24 * 80)
    assert MultiStrategy("kalman_fusion").generate_series(frame).eq(0).all()
    assert build_parser().parse_args(["--strategy", "kalman_fusion"]).strategy == "kalman_fusion"


def test_ensemble_does_not_inherit_global_theta_veto(monkeypatch):
    strategy = MultiStrategy("ensemble_equal")
    monkeypatch.setattr(strategy, "generate_intent", lambda _: Intent(0.2, reason="test", diagnostics={}))
    frame = features(n=100)
    frame["S"] = -0.5
    output = LegacyStrategyAdapter(strategy, RegimeEngine({}), 0.3).generate_intent(frame)
    assert output.target_exposure == 0.2
    assert output.diagnostics["risk_state"] == "SOURCE"


def test_cash_signal_can_exit_losing_position_even_below_hysteresis_threshold():
    class Exit:
        def generate_intent(self, _):
            return Intent(0.0, reason="risk-off", diagnostics={})

    portfolio = PortfolioState(usdt=999, base=0.01, equity=1000, exposure=0.001, avg_entry_price=110)
    bar = MarketBar(ts=1, open=90, high=90, low=89, close=90, volume=1)
    params = EngineParams(fee_rate=0.001, min_notional=0.5, allow_loss_exits=True)
    plan, fill, updated, _ = run_step_simulated(bar, pd.DataFrame(), portfolio, Exit(), params, 0.01, 0.01)
    assert plan.action == "SELL"
    assert fill.status == "filled"
    assert updated.base == 0
    assert updated.realized_pnl_quote < 0


def test_exposure_cap_overrides_hysteresis_and_profit_guard():
    class StayLong:
        def generate_intent(self, _):
            return Intent(0.3, reason="long", diagnostics={})

    portfolio = PortfolioState(usdt=60, base=0.4, equity=100, exposure=0.4, avg_entry_price=200)
    bar = MarketBar(ts=1, open=100, high=102, low=99, close=100, volume=1)
    params = EngineParams(fee_rate=0, min_notional=1, max_exposure=0.3, hyst_floor=0.3)
    plan, fill, updated, _ = run_step_simulated(bar, pd.DataFrame(), portfolio, StayLong(), params, 0.01, 0.01)
    assert plan.action == "SELL"
    assert fill.status == "filled"
    assert updated.exposure <= 0.3 + 1e-3


def test_paper_orchestrator_allows_same_loss_exit(monkeypatch):
    strategy = MultiStrategy("momentum")
    monkeypatch.setattr(strategy, "generate_intent", lambda _: Intent(0.0, reason="exit", diagnostics={}))
    frame = features(n=300)
    balances = {"usdt": 900, "btc": 1, "avg_entry_price": 1000}
    result = compute_step(frame, FeatureConfig(rv_window=10, conc_window=20), RegimeEngine({}),
                          strategy, 0.3, 0.001, balances, mode="paper", min_notional=1)
    assert result.execution["side"] == "sell"
    assert result.equity["btc"] == pytest.approx(0)


def test_new_backtest_approach_uses_core_and_stays_affordable():
    frame = features(n=24 * 90)
    equity, trades, _ = run_backtest(frame, "1h", "momentum", "none", 20, 10, 20,
                                    2, 0.001, 5, 0.3, log=False)
    assert len(trades) > 0
    assert equity.usdt.min() >= 0
    assert equity.position_btc.min() >= 0
