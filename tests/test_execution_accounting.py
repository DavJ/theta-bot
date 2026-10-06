"""Cash conservation, execution timing and spot affordability regressions."""
import pandas as pd
import pytest
import numpy as np

from spot_bot.core import EngineParams, MarketBar, PortfolioState, run_step_simulated
from spot_bot.core.account import SimAccountProvider
from spot_bot.core.engine import simulate_execution
from spot_bot.core.portfolio import apply_fill
from spot_bot.core.types import TradePlan
from spot_bot.strategies.base import Intent


def _plan(qty, limit=None):
    return TradePlan(
        action="BUY" if qty > 0 else "SELL", target_exposure=1.0,
        target_base=abs(qty), delta_base=qty, notional=abs(qty) * 100,
        exec_price_hint=100, reason="test", limit_price=limit,
        order_type="market" if limit is None else "limit",
    )


def test_round_trip_charges_fill_prices_and_fees_once():
    params = EngineParams(fee_rate=0.001, slippage_bps=100, spread_bps=20)
    portfolio = PortfolioState(1000, 0, 1000, 0)
    buy = simulate_execution(_plan(2), 100, params)
    sell = simulate_execution(_plan(-2), 100, params)
    assert buy.avg_price == pytest.approx(101.1)
    assert sell.avg_price == pytest.approx(98.9)
    assert buy.slippage_paid == pytest.approx(2.2)
    final = apply_fill(apply_fill(portfolio, buy), sell)
    expected = 1000 - 2 * 101.1 * 1.001 + 2 * 98.9 * 0.999
    assert final.usdt == pytest.approx(expected)
    assert final.base == 0


def test_untouched_limit_expires_at_close_not_retroactively_at_open():
    bar = MarketBar(0, 100, 112, 100, 110, 10)
    execution = simulate_execution(
        _plan(1, limit=99), 100, EngineParams(slippage_bps=5), bar=bar,
    )
    assert execution.avg_price == pytest.approx(110 * 1.0005)
    assert execution.raw["execution_type"] == "market_timeout"
    assert execution.raw["is_limit"] is False


def test_unfilled_limit_can_be_skipped_without_market_fallback():
    execution = simulate_execution(
        _plan(1, limit=99), 100, EngineParams(limit_timeout_bars=2),
        bar=MarketBar(0, 100, 112, 100, 110, 10),
    )
    assert execution.status == "SKIPPED"
    assert execution.filled_base == 0


def test_full_exposure_spot_buy_respects_cash_fee_reserve_and_step():
    class Buy:
        def generate_intent(self, features):
            return Intent(1.0, "buy", {})

    _, execution, portfolio, _ = run_step_simulated(
        bar=MarketBar(0, 100, 111, 100, 110, 10),
        features_df=pd.DataFrame({"close": [99]}),
        portfolio=PortfolioState(1000, 0, 1000, 0), strategy=Buy(),
        params=EngineParams(fee_rate=0.01, slippage_bps=10,
                            min_usdt_reserve=25, step_size=0.01),
        rv_current=0.01, rv_ref=0.01,
    )
    assert execution.status == "filled"
    assert portfolio.usdt >= 25 - 1e-10
    assert execution.filled_base / 0.01 == pytest.approx(round(execution.filled_base / 0.01))
    assert portfolio.usdt == pytest.approx(
        1000 - execution.filled_base * execution.avg_price - execution.fee_paid
    )


def test_account_preserves_cost_basis_between_bars():
    account = SimAccountProvider(1000)
    account.update_portfolio(PortfolioState(500, 5, 1000, 0.5, 100, 12))
    marked = account.get_portfolio_state(80)
    assert marked.equity == 900
    assert marked.avg_entry_price == 100
    assert marked.realized_pnl_quote == 12


def test_paper_orchestrator_preserves_cost_basis_and_does_not_borrow():
    from spot_bot.features import FeatureConfig
    from spot_bot.regime.regime_engine import RegimeEngine
    from spot_bot.run_live import compute_step

    class Buy:
        def generate_intent(self, features):
            return Intent(1.0, "buy", {})

    price = 100 + np.sin(np.arange(30))
    df = pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                       "close": price, "volume": 10},
                      index=pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC"))
    balances = {"usdt": 1000.0, "btc": 0.0, "realized_pnl_quote": 7.0}
    params = dict(ohlcv_df=df, feature_cfg=FeatureConfig(rv_window=3, conc_window=3, psi_mode="none", psi_window=2),
                  regime_engine=RegimeEngine({"s_off": -1, "s_on": 0}), strategy=Buy(),
                  max_exposure=1.0, fee_rate=0.01, balances=balances, mode="paper", slippage_bps=5)
    first = compute_step(**params)
    assert first.execution["status"] == "filled"
    assert first.equity["usdt"] >= -1e-10
    assert balances["avg_entry_price"] == first.execution["avg_price"]
    assert balances["realized_pnl_quote"] == 7.0
    balances.update(usdt=first.equity["usdt"], btc=first.equity["btc"])
    basis = balances["avg_entry_price"]
    compute_step(**params)
    assert balances["avg_entry_price"] == basis
