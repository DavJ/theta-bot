import pytest

from spot_bot.persist import SQLiteLogger


def test_sqlite_logger_persists_state(tmp_path):
    db_path = tmp_path / "bot.db"
    logger = SQLiteLogger(db_path)

    logger.upsert_bar(ts=1, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0)
    logger.upsert_features(ts=1, rv=0.1, C=0.2, psi=0.3, C_int=0.4, S=0.5)
    logger.upsert_decision(ts=1, risk_state="ON", risk_budget=0.8, reason="test")
    logger.upsert_intent(ts=1, desired_exposure=0.2, reason="intent")
    logger.insert_execution(ts=1, mode="paper", side="buy", qty=0.01, price=20000, fee=0.1, order_id="abc", status="filled", meta={"foo": "bar"})
    logger.upsert_equity(ts=1, equity_usdt=1000.0, btc=0.1, usdt=980.0)

    # idempotent upsert on bar
    logger.upsert_bar(ts=1, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0)

    assert logger.get_last_ts() == 1
    equity = logger.get_latest_equity()
    assert equity["btc"] == pytest.approx(0.1)
    assert equity["usdt"] == pytest.approx(980.0)

    executions = logger.list_executions(limit=1)
    assert executions and executions[0]["meta"] == {"foo": "bar"}

    logger.close()
