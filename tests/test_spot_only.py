"""Reject borrowed exposure and impossible fills before funds or orders change."""
from unittest.mock import MagicMock

import pytest

from spot_bot.core.engine import EngineParams, simulate_execution
from spot_bot.core.portfolio import apply_fill
from spot_bot.core.trade_planner import plan_trade
from spot_bot.core.types import PortfolioState, ExecutionResult
from spot_bot.execution.ccxt_executor import CCXTExecutor, ExecutorConfig


def test_short_switch_is_forbidden_even_after_parameter_mutation():
    with pytest.raises(ValueError, match="Spot-only"):
        EngineParams(allow_short=True)
    state = PortfolioState(1000, 0, 1000, 0)
    with pytest.raises(ValueError, match="Spot-only"):
        plan_trade(state, 100, -.5, 10, allow_short=True)
    params = EngineParams()
    params.allow_short = True
    with pytest.raises(ValueError, match="Spot-only"):
        simulate_execution(plan_trade(state, 100, .5, 10), 100, params, portfolio=state)


@pytest.mark.parametrize("qty,price,fee", [(2, 100, 0), (1, 100, 1), (-2, 100, 0)])
def test_recorded_fill_cannot_create_inventory_or_cash_debt(qty, price, fee):
    state = PortfolioState(100, 1, 200, .5, 100)
    with pytest.raises(ValueError, match="cannot"):
        apply_fill(state, ExecutionResult(qty, price, fee, 0, "filled"))
    assert (state.usdt, state.base) == (100, 1)


def _executor():
    ex = CCXTExecutor(ExecutorConfig(max_notional_per_trade=10000, max_turnover_per_day=20000,
        min_balance_reserve_usdt=25))
    exchange = MagicMock()
    exchange.options = {"defaultType": "spot"}
    exchange.market.return_value = {"spot": True, "contract": False, "base": "BTC", "quote": "USDT"}
    exchange.fetch_balance.return_value = {"free": {"USDT": 1000., "BTC": 2.}}
    exchange.fetch_ticker.return_value = {"bid": 99.99, "ask": 100.01}
    exchange.create_order.return_value = {"id": "fake", "status": "closed", "filled": 1, "average": 100}
    ex.exchange = exchange
    return ex, exchange


@pytest.mark.parametrize("order_type", ["market", "limit"])
@pytest.mark.parametrize("failure", ["short", "cash", "missing", "balance_error", "contract", "margin", "unknown_market", "leveraged_token"])
def test_live_guard_fails_closed_without_submitting_any_order(order_type, failure):
    ex, exchange = _executor()
    side, qty = "buy", 1
    if failure == "short":
        side, qty = "sell", 3
    elif failure == "cash":
        exchange.fetch_balance.return_value = {"free": {"USDT": 100.}}
    elif failure == "missing":
        exchange.fetch_balance.return_value = {"free": {}}
    elif failure == "balance_error":
        exchange.fetch_balance.side_effect = RuntimeError("offline")
    elif failure == "contract":
        exchange.market.return_value.update(spot=False, contract=True)
    elif failure == "margin":
        exchange.options["defaultType"] = "margin"
    elif failure == "unknown_market":
        exchange.market.side_effect = RuntimeError("market unavailable")
    else:
        ex.config.symbol = "BTCUP/USDT"
    call = ex.place_market_order if order_type == "market" else ex.place_limit_maker_order
    result = call(side, qty, 100)
    assert result["status"] == "rejected"
    exchange.create_order.assert_not_called()


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_funded_spot_market_order_uses_free_spot_balance(side):
    ex, exchange = _executor()
    result = ex.place_market_order(side, 1, 100)
    assert result["status"] == "filled"
    assert exchange.create_order.call_count == 1
    assert exchange.fetch_balance.call_args.kwargs == {"params": {"type": "spot"}}


def test_rounding_residual_does_not_invent_meaningful_cash():
    state = PortfolioState(100, 0, 100, 0)
    filled = apply_fill(state, ExecutionResult(1, 100, 1e-12, 0, "filled"))
    assert filled.usdt == 0
    assert filled.base == 1


def test_sell_can_reduce_spot_risk_when_quote_cash_is_below_reserve():
    ex, exchange = _executor()
    exchange.fetch_balance.return_value = {"free": {"USDT": 0., "BTC": 2.}}
    assert ex.place_market_order("sell", 1, 100)["status"] == "filled"
    assert exchange.create_order.call_count == 1


@pytest.mark.parametrize("field", ["fee_rate", "maker_fee_rate", "taker_fee_rate"])
def test_unknown_fee_cannot_authorize_a_spot_order(field):
    ex, exchange = _executor()
    setattr(ex.config, field, float("nan"))
    assert ex.place_market_order("buy", 1, 100)["status"] == "rejected"
    exchange.create_order.assert_not_called()
