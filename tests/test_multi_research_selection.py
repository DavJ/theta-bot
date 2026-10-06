from scripts.research_multi_strategy import CANDIDATES, select_on_development
from spot_bot.evaluate import EvaluationConfig


def test_profit_is_maximized_among_drawdown_eligible_candidates():
    development = {c.name: {"net_pnl": -1, "maxDD": -0.01} for c in CANDIDATES}
    development["legacy_theta"] = {"net_pnl": 10, "maxDD": -0.1}
    development["kalman_fusion"] = {"net_pnl": 100, "maxDD": -0.3}
    assert select_on_development(development, EvaluationConfig()).name == "legacy_theta"


def test_cash_is_selected_when_all_candidates_lose():
    development = {c.name: {"net_pnl": -1, "maxDD": -0.01} for c in CANDIDATES}
    assert select_on_development(development, EvaluationConfig()) is None
