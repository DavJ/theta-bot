"""Selection isolation, source validation and fail-closed screening."""
from dataclasses import replace

import pandas as pd
import pytest

from spot_bot import evaluate as ev


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "prices.csv"
    pd.read_csv(ev.BUNDLED_DATA).iloc[:1000].to_csv(path, index=False)
    return path


def _config():
    return ev.EvaluationConfig(rv_window=10, psi_window=20, conc_window=20)


def _fake_backtest(df, strategy, config, start=None):
    net = 10 if start is None and strategy == "meanrev" else (1 if start is None else -5)
    summary = {"net_pnl": net, "total_return": net / 1000, "sharpe": -1,
               "maxDD": -0.05, "trades_count": 50, "fees_paid_total": 1}
    return pd.DataFrame(), pd.DataFrame(), summary


def test_selection_is_frozen_before_holdout_and_losing_choice_fails(csv_path, monkeypatch):
    calls = []

    def capture(*args):
        calls.append((args[1], len(args[0]), args[3] if len(args) > 3 else None))
        return _fake_backtest(*args)

    monkeypatch.setattr(ev, "_backtest", capture)
    report, _, _ = ev.evaluate(csv_path, _config(), as_of="2026-10-06T06:00Z")
    assert [item[0] for item in calls[:4]] == list(ev.STRATEGIES)
    assert all(item[1] == 600 for item in calls[:4])
    assert all(item[0] == "meanrev" for item in calls[4:])
    assert report["selected_strategy"] == "meanrev"
    assert not report["gates"]["holdout_net_positive"]
    assert not report["gates"]["data_current"]
    assert not report["gates"]["market_data_provenance"]
    assert not report["paper_screen_passed"]
    assert not report["live_eligible"]


@pytest.mark.parametrize("damage, expected", [
    ("gap", "Missing candles"), ("nan", "nonfinite"),
    ("range", "OHLC ranges"), ("duplicate", "duplicate timestamps"),
])
def test_market_data_errors_are_rejected(csv_path, damage, expected):
    df = pd.read_csv(csv_path)
    if damage == "gap":
        df = df.drop(index=100)
    elif damage == "nan":
        df.loc[100, "close"] = float("nan")
    elif damage == "range":
        df.loc[100, "high"] = df.loc[100, "low"] - 1
    else:
        df.loc[100, "timestamp"] = df.loc[99, "timestamp"]
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match=expected):
        ev.evaluate(csv_path, _config(), as_of="2026-10-06T06:00Z")


def test_unclosed_candles_are_rejected(csv_path):
    with pytest.raises(ValueError, match="unclosed/future"):
        ev.evaluate(csv_path, _config(), as_of="2024-04-01T00:30Z")


def test_archive_label_requires_matching_provenance(csv_path):
    with pytest.raises(ValueError, match="manifest is missing"):
        ev.evaluate(csv_path, _config(), as_of="2026-10-06T06:00Z", data_source="binance_archive")


@pytest.mark.parametrize("field,value", [("fee_rate", -0.001), ("slippage_bps", -1),
                                          ("max_exposure", 1.1), ("initial_usdt", 0)])
def test_invalid_screening_parameters_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(_config(), **{field: value}).validate()


def test_require_paper_pass_writes_evidence_and_returns_failure(csv_path, tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "_backtest", _fake_backtest)
    # Exercise the actual writer/exit status with a failed, already-computed report.
    report, equity, trades = ev.evaluate(csv_path, _config(), as_of="2026-10-06T06:00Z")
    monkeypatch.setattr(ev, "evaluate", lambda *args, **kwargs: (report, equity, trades))
    out = tmp_path / "evidence"
    assert ev.main(["--csv", str(csv_path), "--out", str(out), "--require-paper-pass"]) == 2
    assert (out / "summary.json").is_file()
    assert "Paper screening: **FAIL**" in (out / "report.md").read_text()
