"""Tests for LSTMKalmanStrategy."""
import numpy as np
import pandas as pd
import pytest

from spot_bot.strategies.lstm_kalman import (
    LSTMKalmanStrategy,
    LSTMKalmanParams,
    _NumpyLSTMCell,
    _run_local_linear_kalman,
    _sigmoid,
    _sigmoid_to_signed,
    _get_feature_col,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _synthetic_features(n: int = 120, seed: int = 0, with_biquat: bool = True) -> pd.DataFrame:
    """Build synthetic OHLCV + feature DataFrame that mimics spot_bot feature pipeline output."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 30_000.0 + np.cumsum(rng.normal(0, 100, size=n))
    rv = np.abs(rng.normal(0.02, 0.005, size=n))
    data = {"close": close, "rv": rv}
    if with_biquat:
        # Biquaternion-inspired features from unified-biquaternion-theory feature pipeline
        data["C"] = 0.6 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, n))
        data["psi"] = (np.linspace(0, 4, n) % 1.0)
        data["S"] = (np.linspace(0, 2, n) % 1.0)
        data["risk_budget"] = np.clip(rng.normal(0.8, 0.1, size=n), 0.0, 1.0)
    return pd.DataFrame(data, index=idx)


# ──────────────────────────────────────────────────────────────────────────────
# Unit: NumpyLSTMCell
# ──────────────────────────────────────────────────────────────────────────────

def test_lstm_cell_output_shapes():
    cell = _NumpyLSTMCell(input_size=7, hidden_size=32, seed=0)
    h, c = cell.zero_state()
    assert h.shape == (32,)
    assert c.shape == (32,)
    x = np.random.default_rng(1).normal(size=7)
    h2, c2 = cell.step(x, h, c)
    assert h2.shape == (32,)
    assert c2.shape == (32,)
    proj = cell.project(h2)
    assert np.isfinite(proj)


def test_lstm_cell_deterministic():
    cell = _NumpyLSTMCell(input_size=7, hidden_size=16, seed=42)
    x = np.ones(7)
    h, c = cell.zero_state()
    h1, c1 = cell.step(x, h, c)
    # Re-run from zero state
    h0, c0 = cell.zero_state()
    h2, c2 = cell.step(x, h0, c0)
    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(c1, c2)


def test_lstm_cell_output_bounded():
    """h_t = o * tanh(c) so ||h||_inf <= 1; project gives tanh-squashed scalar."""
    cell = _NumpyLSTMCell(input_size=7, hidden_size=32, seed=0)
    rng = np.random.default_rng(3)
    h, c = cell.zero_state()
    for _ in range(50):
        x = rng.normal(size=7)
        h, c = cell.step(x, h, c)
    # h values must be in (-1, 1) (tanh output)
    assert np.all(np.abs(h) <= 1.0 + 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Unit: _run_local_linear_kalman
# ──────────────────────────────────────────────────────────────────────────────

def test_kalman_returns_finite():
    prices = pd.Series([100.0 + i * 0.5 for i in range(50)])
    level, trend, innov_var = _run_local_linear_kalman(prices, q_level=1e-4, q_trend=1e-6, r=1e-3)
    assert np.isfinite(level)
    assert np.isfinite(trend)
    assert innov_var > 0.0


def test_kalman_level_tracks_flat_series():
    prices = pd.Series([100.0] * 200)
    level, trend, innov_var = _run_local_linear_kalman(prices, q_level=1e-4, q_trend=1e-6, r=1e-3)
    assert abs(level - 100.0) < 1.0
    assert abs(trend) < 0.1


# ──────────────────────────────────────────────────────────────────────────────
# Unit: _sigmoid
# ──────────────────────────────────────────────────────────────────────────────

def test_sigmoid_bounds():
    x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    s = _sigmoid(x)
    assert np.all(s >= 0.0)
    assert np.all(s <= 1.0)
    assert abs(s[2] - 0.5) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: generate_intent
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_intent_basic():
    strat = LSTMKalmanStrategy(min_bars=10)
    df = _synthetic_features(n=80)
    intent = strat.generate_intent(df)
    assert np.isfinite(intent.desired_exposure)
    assert 0.0 <= intent.desired_exposure <= 1.0


def test_generate_intent_empty_returns_zero():
    strat = LSTMKalmanStrategy()
    intent = strat.generate_intent(pd.DataFrame())
    assert intent.desired_exposure == 0.0


def test_generate_intent_insufficient_history():
    strat = LSTMKalmanStrategy(min_bars=50)
    df = _synthetic_features(n=10)
    intent = strat.generate_intent(df)
    assert intent.desired_exposure == 0.0


def test_generate_intent_deterministic():
    strat = LSTMKalmanStrategy(min_bars=10)
    df = _synthetic_features(n=60)
    intent1 = strat.generate_intent(df)
    intent2 = strat.generate_intent(df)
    assert intent1.desired_exposure == intent2.desired_exposure


def test_generate_intent_diagnostics_keys():
    strat = LSTMKalmanStrategy(min_bars=10)
    df = _synthetic_features(n=60)
    intent = strat.generate_intent(df)
    for key in ("level", "trend", "kalman_z", "kalman_signal", "lstm_signal", "combined", "risk_budget"):
        assert key in intent.diagnostics, f"Missing diagnostic key: {key}"


def test_generate_intent_without_biquat_features():
    """Should work even when C/psi/S are absent (defaults to 0/0.5/0.5)."""
    strat = LSTMKalmanStrategy(min_bars=10)
    df = _synthetic_features(n=60, with_biquat=False)
    intent = strat.generate_intent(df)
    assert np.isfinite(intent.desired_exposure)
    assert 0.0 <= intent.desired_exposure <= 1.0


def test_generate_intent_risk_budget_zero():
    """risk_budget=0 must clamp exposure to 0."""
    strat = LSTMKalmanStrategy(min_bars=10)
    df = _synthetic_features(n=60)
    df["risk_budget"] = 0.0
    intent = strat.generate_intent(df)
    assert intent.desired_exposure == 0.0


def test_generate_intent_kalman_weight_one():
    """kalman_weight=1 → pure Kalman signal."""
    strat_k = LSTMKalmanStrategy(min_bars=10, kalman_weight=1.0, lstm_seed=0)
    df = _synthetic_features(n=80)
    intent = strat_k.generate_intent(df)
    diag = intent.diagnostics
    expected = float(np.clip(diag["kalman_signal"] * 1.0 * diag["risk_budget"], 0.0, 1.0))
    assert abs(intent.desired_exposure - expected) < 1e-9


def test_generate_intent_kalman_weight_zero():
    """kalman_weight=0 → pure LSTM signal."""
    strat_l = LSTMKalmanStrategy(min_bars=10, kalman_weight=0.0, lstm_seed=0)
    df = _synthetic_features(n=80)
    intent = strat_l.generate_intent(df)
    diag = intent.diagnostics
    expected = float(np.clip(diag["lstm_signal"] * 1.0 * diag["risk_budget"], 0.0, 1.0))
    assert abs(intent.desired_exposure - expected) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: generate_series
# ──────────────────────────────────────────────────────────────────────────────

def test_generate_series_length():
    strat = LSTMKalmanStrategy(min_bars=10)
    df = _synthetic_features(n=100)
    series = strat.generate_series(df)
    assert len(series) == len(df)


def test_generate_series_bounded():
    strat = LSTMKalmanStrategy(min_bars=10, emax=1.0)
    df = _synthetic_features(n=100)
    series = strat.generate_series(df)
    assert series.notna().all()
    assert (series >= 0.0).all()
    assert (series <= 1.0 + 1e-9).all()


def test_generate_series_min_bars_prefix_zero():
    min_bars = 30
    strat = LSTMKalmanStrategy(min_bars=min_bars)
    df = _synthetic_features(n=80)
    series = strat.generate_series(df)
    assert (series.iloc[: min_bars - 1] == 0.0).all()


def test_generate_series_deterministic():
    strat = LSTMKalmanStrategy(min_bars=10)
    df = _synthetic_features(n=60)
    s1 = strat.generate_series(df)
    s2 = strat.generate_series(df)
    pd.testing.assert_series_equal(s1, s2)


def test_generate_series_empty():
    strat = LSTMKalmanStrategy()
    s = strat.generate_series(pd.DataFrame())
    assert len(s) == 0


def test_generate_series_external_risk_budgets():
    strat = LSTMKalmanStrategy(min_bars=10)
    df = _synthetic_features(n=80, with_biquat=False)
    rb = pd.Series(0.5, index=df.index)
    s_half = strat.generate_series(df, risk_budgets=rb)
    rb_full = pd.Series(1.0, index=df.index)
    s_full = strat.generate_series(df, risk_budgets=rb_full)
    # With half risk budget, exposure should be ≤ full risk budget exposure everywhere
    assert (s_half <= s_full + 1e-9).all()


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: registration in __init__.py
# ──────────────────────────────────────────────────────────────────────────────

def test_strategy_importable_from_package():
    from spot_bot.strategies import LSTMKalmanStrategy as S
    assert S is LSTMKalmanStrategy
