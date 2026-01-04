import pandas as pd
import pytest

from spot_bot.regime.regime_engine import RegimeEngine


def test_risk_budget_clipped():
    engine = RegimeEngine({"s_off": 0.0, "s_on": 1.0, "s_budget_low": 0.0, "s_budget_high": 1.0})
    df = pd.DataFrame({"C": [0.1], "S": [1.5], "rv": [0.0]})

    decision = engine.decide(df)
    assert decision.risk_budget == 1.0


def test_regime_transitions():
    config = {"s_off": 0.2, "s_on": 0.6, "rv_off": 0.5, "rv_reduce": 0.3}
    engine = RegimeEngine(config)

    rows = [
        (0.1, 0.1, 0.2, "OFF"),
        (0.4, 0.1, 0.2, "REDUCE"),
        (0.8, 0.1, 0.2, "ON"),
        (0.8, 0.1, 0.6, "OFF"),
    ]

    for s_val, c_val, rv_val, expected in rows:
        df = pd.DataFrame({"C": [c_val], "S": [s_val], "rv": [rv_val]})
        decision = engine.decide(df)
        assert decision.risk_state == expected


def test_missing_columns_raise():
    engine = RegimeEngine({})
    df_missing = pd.DataFrame({"S": [0.1], "rv": [0.1]})

    with pytest.raises(ValueError):
        engine.decide(df_missing)
