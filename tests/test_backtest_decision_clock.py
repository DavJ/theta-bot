"""Structural causality tests: future candles cannot change past decisions."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spot_bot.backtest import fast_backtest as fb
from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy
from spot_bot.strategies.lstm_kalman import LSTMKalmanStrategy


DATA = Path(__file__).resolve().parents[1] / "data/BTCUSDT_1H_real.csv.gz"


def _run(df, strategy="kalman", psi_mode="none", **kwargs):
    return fb.run_backtest(
        df, "1h", strategy, psi_mode, 20, 10, 20, 2.0,
        0.001, 5, 0.3, log=False, **kwargs,
    )


@pytest.mark.parametrize("strategy", ["meanrev", "kalman", "kalman_mr_dual", "lstm_kalman"])
@pytest.mark.parametrize("psi_mode", ["none", "scale_phase"])
def test_append_future_candles_leaves_previous_equity_and_trades_identical(strategy, psi_mode):
    df = pd.read_csv(DATA).iloc[:450]
    prefix_equity, prefix_trades, _ = _run(df.iloc[:300], strategy, psi_mode)
    full_equity, full_trades, _ = _run(df, strategy, psi_mode)
    cutoff = pd.to_datetime(df.iloc[299]["timestamp"], unit="ms", utc=True)
    pd.testing.assert_frame_equal(
        prefix_equity, full_equity.loc[full_equity.timestamp <= cutoff].reset_index(drop=True),
    )
    if not prefix_trades.empty:
        pd.testing.assert_frame_equal(
            prefix_trades, full_trades.loc[full_trades.timestamp <= cutoff].reset_index(drop=True),
        )


def test_current_close_does_not_affect_opening_order_size_or_hysteresis(monkeypatch):
    df = pd.read_csv(DATA).iloc[:300].copy()
    changed = df.copy()
    changed.loc[200, "close"] *= 1.05
    changed.loc[200, "high"] = max(changed.loc[200, "high"], changed.loc[200, "close"])
    monkeypatch.setattr(fb, "_compute_intents_with_regime", lambda f, *args: pd.Series(0.3, index=f.index))
    original = fb.run_step_simulated
    records = []

    def capture(**kwargs):
        result = original(**kwargs)
        plan, _, _, diagnostics = result
        records.append((kwargs["bar"].ts, plan.delta_base, diagnostics["delta_e_min"]))
        return result

    monkeypatch.setattr(fb, "run_step_simulated", capture)
    _run(df)
    first = pd.DataFrame(records, columns=["ts", "qty", "threshold"]).set_index("ts")
    records.clear()
    _run(changed)
    second = pd.DataFrame(records, columns=["ts", "qty", "threshold"]).set_index("ts")
    ts = int(df.loc[200, "timestamp"])
    pd.testing.assert_series_equal(first.loc[ts], second.loc[ts])


@pytest.mark.parametrize("strategy", [MeanRevDualKalmanStrategy(), LSTMKalmanStrategy()])
def test_regime_budget_is_applied_once_and_on_the_decision_bar(strategy, monkeypatch):
    features = pd.DataFrame({"close": [100, 99, 98, 97]})
    budget = pd.Series([0, 0.25, 0.5, 0.75])

    def generate(features, risk_budgets, **kwargs):
        return risk_budgets * 0.8

    monkeypatch.setattr(strategy, "generate_series", generate)
    actual = fb._compute_intents_with_regime(features, strategy, pd.Series("ON", index=features.index), budget, 1.0)
    np.testing.assert_allclose(actual, [0, 0.2, 0.4, 0.6])


def test_sharpe_has_one_annualization_and_includes_first_bar_cost():
    equity, _, summary = _run(pd.read_csv(DATA).iloc[:450])
    values = equity.equity.to_numpy()
    returns = values / np.r_[1000.0, values[:-1]] - 1
    expected = returns.mean() / returns.std(ddof=0) * np.sqrt(8760)
    assert summary["sharpe"] == pytest.approx(expected)


def test_normalization_does_not_change_caller_data():
    df = pd.read_csv(DATA).iloc[:300]
    before = df.copy(deep=True)
    _run(df)
    pd.testing.assert_frame_equal(df, before)


def test_holdout_uses_prior_history_without_trading_it():
    df = pd.read_csv(DATA).iloc[:450]
    cutoff = pd.to_datetime(df.iloc[300]["timestamp"], unit="ms", utc=True)
    equity, trades, _ = _run(df, evaluation_start=cutoff)
    assert equity.timestamp.iloc[0] == cutoff
    assert len(equity) == 150
    if not trades.empty:
        assert trades.timestamp.min() >= cutoff
    assert equity.usdt.min() >= -1e-10


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unsupported strategy"):
        _run(pd.read_csv(DATA).iloc[:300], "typo")
