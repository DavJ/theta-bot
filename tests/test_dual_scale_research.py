"""Holdout comparisons cannot reselect a predeclared strategy."""
import numpy as np
import pandas as pd
import pytest

from scripts import research_dual_scale as research
from spot_bot.evaluate import EvaluationConfig, load_market_data


def frame():
    times = pd.date_range("2024-12-01", "2026-01-02", freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp": times, "open": 100, "high": 101,
                         "low": 99, "close": 100, "volume": 10})


@pytest.mark.parametrize("all_development_loses", [False, True])
def test_selection_is_frozen_and_nonpositive_development_abstains(monkeypatch, all_development_loses):
    calls = []

    def run(data, candidate, config, start=None):
        calls.append((candidate.name, start, data.timestamp.max()))
        if start is None:
            net = {"legacy_dual": 5, "log_vol_dual": 10, "log_vol_no_conf": -1}[candidate.name]
            if all_development_loses:
                net -= 20
        else:
            # A different candidate wins validation; it must not be selected.
            net = 100 if candidate.name == "legacy_dual" else -5
        summary = {"net_pnl": net, "total_return": net / 1000, "maxDD": -0.01,
                   "sharpe": 2, "mean_exposure": 0.04}
        return pd.DataFrame(), pd.DataFrame(), summary

    monkeypatch.setattr(research, "run_candidate", run)
    report, _, _ = research.research(frame())
    assert all(call[1] is None for call in calls[:3])
    assert all(call[2] < research.DEVELOPMENT_END for call in calls[:3])
    assert report["frozen_candidate"] == "log_vol_dual"
    assert report["development_paper_candidate"] == ("cash" if all_development_loses else "log_vol_dual")
    assert not report["gates"]["validation_net_positive"]
    assert not report["research_screen_passed"]
    assert not report["live_eligible"]


def test_default_loader_rejects_gaps_and_research_opt_in_preserves_them(tmp_path):
    path = tmp_path / "data.csv"
    original = frame().iloc[:100].drop(index=20)
    original.to_csv(path, index=False)
    config = EvaluationConfig()
    with pytest.raises(ValueError, match="Missing candles"):
        load_market_data(path, config, pd.Timestamp("2026-10-06", tz="UTC"))
    loaded = load_market_data(path, config, pd.Timestamp("2026-10-06", tz="UTC"), allow_gaps=True)
    assert len(loaded) == len(original)
    assert pd.Timestamp("2024-12-01T20:00Z") not in loaded.timestamp.to_list()
