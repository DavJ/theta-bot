"""Integration test for Mellin cepstrum modes in feature pipeline."""

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


def test_mellin_cepstrum_mode_produces_valid_features():
    """Test that mellin_cepstrum mode produces valid features."""
    ohlcv = _synthetic_ohlcv(300)
    cfg = FeatureConfig(
        psi_mode="mellin_cepstrum",
        psi_window=64,
        mellin_grid_n=128,
        mellin_sigma=0.0,
        psi_min_bin=2,
        psi_max_frac=0.25,
        psi_phase_agg="peak",
    )
    
    features = compute_features(ohlcv, cfg)
    
    assert not features.empty
    assert "psi" in features.columns
    assert "psi_mode" in features.columns
    assert features["psi_mode"].iloc[-1] == "mellin_cepstrum"
    
    # Check that psi values are in valid range [0, 1)
    psi_vals = features["psi"].dropna()
    assert not psi_vals.empty
    assert ((psi_vals >= 0.0) & (psi_vals < 1.0)).all()


def test_mellin_complex_cepstrum_mode_produces_valid_features():
    """Test that mellin_complex_cepstrum mode produces valid features."""
    ohlcv = _synthetic_ohlcv(300)
    cfg = FeatureConfig(
        psi_mode="mellin_complex_cepstrum",
        psi_window=64,
        mellin_grid_n=128,
        mellin_sigma=0.0,
        mellin_detrend_phase=True,
        psi_min_bin=2,
        psi_max_frac=0.25,
        psi_phase_agg="cmean",
        psi_phase_power=1.5,
    )
    
    features = compute_features(ohlcv, cfg)
    
    assert not features.empty
    assert "psi" in features.columns
    assert "psi_mode" in features.columns
    assert features["psi_mode"].iloc[-1] == "mellin_complex_cepstrum"
    
    # Check that psi values are in valid range [0, 1)
    psi_vals = features["psi"].dropna()
    assert not psi_vals.empty
    assert ((psi_vals >= 0.0) & (psi_vals < 1.0)).all()
    
    # Check debug columns exist
    assert "psi_n_star" in features.columns
    assert "psi_c_real" in features.columns
    assert "psi_c_imag" in features.columns


def test_default_cepstrum_mode_still_works():
    """Test that default cepstrum mode still works (backward compatibility)."""
    ohlcv = _synthetic_ohlcv(300)
    cfg = FeatureConfig()  # Default settings
    
    features = compute_features(ohlcv, cfg)
    
    assert not features.empty
    assert "psi" in features.columns
    assert features["psi_mode"].iloc[-1] == "cepstrum"
    
    psi_vals = features["psi"].dropna()
    assert not psi_vals.empty
    assert ((psi_vals >= 0.0) & (psi_vals < 1.0)).all()


def test_all_psi_modes_produce_consistent_output_structure():
    """Test that all psi modes produce consistent feature structure."""
    ohlcv = _synthetic_ohlcv(300)
    modes = ["cepstrum", "complex_cepstrum", "mellin_cepstrum", "mellin_complex_cepstrum"]
    
    base_columns = {"timestamp", "close", "rv", "phi", "cos_phi", "sin_phi", "C", "psi", "C_int", "S", "psi_mode"}
    
    for mode in modes:
        cfg = FeatureConfig(psi_mode=mode, psi_window=64)
        features = compute_features(ohlcv, cfg)
        
        assert not features.empty, f"Empty features for mode {mode}"
        
        # All modes should have base columns
        for col in base_columns:
            assert col in features.columns, f"Missing {col} in mode {mode}"
        
        # Check psi values
        psi_vals = features["psi"].dropna()
        assert not psi_vals.empty, f"No valid psi values for mode {mode}"
        assert ((psi_vals >= 0.0) & (psi_vals < 1.0)).all(), f"Invalid psi range for mode {mode}"


def test_mellin_with_different_sigmas():
    """Test that different sigma values produce different results."""
    ohlcv = _synthetic_ohlcv(300)
    
    cfg1 = FeatureConfig(psi_mode="mellin_cepstrum", psi_window=64, mellin_sigma=0.0)
    cfg2 = FeatureConfig(psi_mode="mellin_cepstrum", psi_window=64, mellin_sigma=0.5)
    
    features1 = compute_features(ohlcv, cfg1)
    features2 = compute_features(ohlcv, cfg2)
    
    psi1 = features1["psi"].dropna()
    psi2 = features2["psi"].dropna()
    
    assert not psi1.empty
    assert not psi2.empty
    
    # Different sigma should produce different results (though maybe not all values)
    # At least some values should differ
    assert not (psi1 == psi2).all()
