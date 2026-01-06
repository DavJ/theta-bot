import numpy as np
import pandas as pd

from spot_bot.strategies.meanrev_dual_kalman import MeanRevDualKalmanStrategy
from theta_bot_averaging.filters.dual_kalman import RegimeKalman1D, circle_dist


def _synthetic_features(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100 + np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.5, size=n)
    C = 0.6 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, n))
    psi = (np.linspace(0, 2, n) % 1.0)
    rv = np.abs(rng.normal(0.02, 0.01, size=n))
    return pd.DataFrame({"close": close, "C": C, "psi": psi, "rv": rv}, index=idx)


def test_regime_kalman_smoothing_deterministic():
    rng = np.random.default_rng(42)
    z_vals = rng.normal(0.0, 1.0, size=200)
    filt1 = RegimeKalman1D(q=0.05, r=0.2)
    filt2 = RegimeKalman1D(q=0.05, r=0.2)
    r1 = [filt1.step(z) for z in z_vals]
    r2 = [filt2.step(z) for z in z_vals]
    assert np.allclose(r1, r2)
    assert np.std(r1) < np.std(z_vals)


def test_circle_dist_range_and_bounds():
    assert 0.0 <= circle_dist(0.1, 0.9) <= 0.5
    assert 0.0 <= circle_dist(0.25, 0.75) <= 0.5
    assert np.isnan(circle_dist(np.nan, 0.2))


def test_strategy_sane_scale_and_exposure():
    feat = _synthetic_features(n=80, seed=7)
    strat = MeanRevDualKalmanStrategy(emax=0.5, s_min=0.3, s_max=2.0, sigma_window=12)
    intent = strat.generate_intent(feat)
    assert not np.isnan(intent.desired_exposure)
    assert 0.0 <= intent.desired_exposure <= 0.5
    assert 0.3 <= intent.diagnostics.get("scale", 0.0) <= 2.0
    assert not np.isnan(intent.diagnostics.get("sigma", np.nan))


def test_strategy_smoke_backtest_series():
    feat = _synthetic_features(n=120, seed=2)
    strat = MeanRevDualKalmanStrategy(emax=1.0, sigma_window=10)
    exposures = strat.generate_series(feat, apply_budget=False)
    assert len(exposures) == len(feat)
    assert exposures.notna().all()
    assert exposures.max() <= 1.0 + 1e-9
