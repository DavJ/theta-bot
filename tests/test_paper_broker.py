import pytest

from spot_bot.live import PaperBroker


def test_buy_then_sell_and_fee_applied():
    broker = PaperBroker(initial_usdt=1000.0, fee_rate=0.001, min_notional=10.0)
    price = 20000.0

    buy = broker.trade_to_target_btc(target_btc=0.02, price=price)
    assert buy["status"] in ("filled", "partial")
    balances_after_buy = broker.balances()
    assert balances_after_buy["btc"] > 0.0
    assert broker.equity(price) < 1000.0  # fee reduces equity

    sell = broker.trade_to_target_btc(target_btc=0.0, price=price)
    assert sell["status"] in ("filled", "partial")
    assert broker.balances()["btc"] == pytest.approx(0.0, abs=1e-8)


def test_cannot_exceed_balance_and_min_notional():
    broker = PaperBroker(initial_usdt=20.0, fee_rate=0.001, min_notional=10.0)
    res = broker.trade_to_target_btc(target_btc=0.01, price=50000.0)
    assert res["status"] in ("filled", "partial", "rejected")
    balances = broker.balances()
    assert balances["usdt"] >= 0.0
    assert balances["btc"] <= 0.01 + 1e-8

    too_small = broker.trade_to_target_btc(target_btc=0.00001, price=20000.0)
    assert too_small["status"] == "rejected"


def test_trade_reaches_target_within_tolerance():
    broker = PaperBroker(initial_usdt=2000.0, fee_rate=0.0, min_notional=10.0)
    price = 25000.0
    target_btc = 0.05

    res = broker.trade_to_target_btc(target_btc=target_btc, price=price)
    assert res["status"] in ("filled", "partial")
    assert broker.balances()["btc"] == pytest.approx(target_btc, rel=1e-6, abs=1e-6)
