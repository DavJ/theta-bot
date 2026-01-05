"""Integration test for scale-phase mode in feature pipeline."""

import numpy as np
import pandas as pd
import pytest

from spot_bot.features import FeatureConfig, compute_features


def _synthetic_ohlcv(n: int = 500) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 20000 + np.linspace(0, 500, n)
    noise = np.sin(np.linspace(0, 6.28, n)) * 50
    close = base + noise
    
    np.random.seed(42)
    open_noise = np.random.normal(0, 0.0005, size=n)
    spread_noise = np.abs(np.random.normal(0, 0.0005, size=n))
    
    open_ = close * (1 + open_noise)
    high = np.maximum(open_, close) * (1 + spread_noise)
    low = np.minimum(open_, close) * (1 - spread_noise)
    volume = np.full(n, 1.0)
    
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_scale_phase_mode_produces_valid_features():
    """Test that scale_phase mode produces valid features."""
    ohlcv = _synthetic_ohlcv(300)
    cfg = FeatureConfig(psi_mode="scale_phase", psi_window=64, base=10.0)

    features = compute_features(ohlcv, cfg)

    assert not features.empty
    assert "psi" in features.columns
    assert "psi_mode" in features.columns
    assert features["psi_mode"].iloc[-1] == "scale_phase"

    psi_vals = features["psi"].dropna()
    assert not psi_vals.empty
    assert ((psi_vals >= 0.0) & (psi_vals < 1.0)).all()


def test_none_mode_disables_internal_phase():
    """Psi mode none should return NaN psi values."""
    ohlcv = _synthetic_ohlcv(128)
    cfg = FeatureConfig(psi_mode="none", psi_window=32)

    features = compute_features(ohlcv, cfg)
    assert "psi" in features.columns
    assert features["psi_mode"].iloc[-1] == "none"
    assert features["psi"].isna().all()


def test_scale_phase_matches_direct_computation():
    """Pipeline psi should match direct compute_scale_phase."""
    from theta_features.scale_phase import compute_scale_phase

    ohlcv = _synthetic_ohlcv(200)
    cfg = FeatureConfig(psi_mode="scale_phase", psi_window=32, base=5.0)

    features = compute_features(ohlcv, cfg)
    expected = compute_scale_phase(features["rv"], window=cfg.psi_window, base=cfg.base)

    pd.testing.assert_series_equal(features["psi"], expected, check_names=False)
