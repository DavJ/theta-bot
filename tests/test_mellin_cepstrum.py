import math

import numpy as np
import pandas as pd
import pytest

from theta_features.cepstrum import (
    mellin_cepstral_phase,
    mellin_complex_cepstral_phase,
    mellin_transform,
    rolling_mellin_cepstral_phase,
    rolling_mellin_complex_cepstral_phase,
)


def test_mellin_transform_basic():
    """Test basic Mellin transform computation."""
    x = np.linspace(1.0, 2.0, 64)
    x_m = mellin_transform(x, grid_n=128, sigma=0.0)
    assert x_m.shape == (128,)
    assert np.iscomplexobj(x_m)


def test_mellin_transform_with_sigma():
    """Test Mellin transform with non-zero sigma."""
    x = np.sin(np.linspace(0, 2 * np.pi, 64)) + 2.0
    x_m1 = mellin_transform(x, grid_n=64, sigma=0.0)
    x_m2 = mellin_transform(x, grid_n=64, sigma=0.5)
    # Different sigma should give different results
    assert not np.allclose(x_m1, x_m2)


def test_mellin_cepstral_phase_constant_signal():
    """Test that constant signal produces valid phase."""
    x = np.ones(32, dtype=float)
    psi = mellin_cepstral_phase(x, grid_n=64, min_bin=2, max_frac=0.25)
    assert 0.0 <= psi < 1.0


def test_mellin_cepstral_phase_is_deterministic():
    """Test deterministic behavior of Mellin cepstral phase."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=64) + 2.0
    psi1 = mellin_cepstral_phase(x, grid_n=128, sigma=0.0, phase_agg="peak")
    psi2 = mellin_cepstral_phase(x, grid_n=128, sigma=0.0, phase_agg="peak")
    assert not math.isnan(psi1)
    assert psi1 == pytest.approx(psi2)
    assert 0.0 <= psi1 < 1.0


def test_mellin_complex_cepstral_phase_with_detrending():
    """Test complex Mellin cepstral phase with and without detrending."""
    rng = np.random.default_rng(123)
    x = rng.normal(size=64) + 3.0
    psi1 = mellin_complex_cepstral_phase(x, grid_n=128, detrend_phase=True)
    psi2 = mellin_complex_cepstral_phase(x, grid_n=128, detrend_phase=False)
    assert 0.0 <= psi1 < 1.0
    assert 0.0 <= psi2 < 1.0
    # They may differ due to detrending
    assert not math.isnan(psi1)
    assert not math.isnan(psi2)


def test_mellin_cepstral_phase_cmean_aggregation():
    """Test circular mean aggregation for phase extraction."""
    rng = np.random.default_rng(456)
    x = rng.normal(size=64) + 2.0
    psi_peak = mellin_cepstral_phase(x, grid_n=128, phase_agg="peak")
    psi_cmean = mellin_cepstral_phase(x, grid_n=128, phase_agg="cmean", phase_power=1.5)
    assert 0.0 <= psi_peak < 1.0
    assert 0.0 <= psi_cmean < 1.0


def test_rolling_mellin_cepstral_phase_basic():
    """Test rolling Mellin cepstral phase computation."""
    idx = pd.date_range("2024-01-01", periods=100, freq="h")
    series = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 100)) + 3.0, index=idx)
    res = rolling_mellin_cepstral_phase(series, window=32, grid_n=64)
    vals = res.dropna()
    assert not vals.empty
    assert ((vals >= 0.0) & (vals < 1.0)).all()


def test_rolling_mellin_complex_cepstral_phase_with_debug():
    """Test rolling Mellin complex cepstral phase with debug outputs."""
    idx = pd.date_range("2024-01-01", periods=80, freq="h")
    series = pd.Series(np.linspace(1.0, 2.0, 80), index=idx)
    psi, debug = rolling_mellin_complex_cepstral_phase(
        series, window=24, grid_n=64, detrend_phase=True, return_debug=True
    )
    vals = psi.dropna()
    assert not vals.empty
    assert ((vals >= 0.0) & (vals < 1.0)).all()
    assert "psi_n_star" in debug.columns
    assert "psi_c_real" in debug.columns
    assert "psi_c_imag" in debug.columns


def test_mellin_phase_handles_empty_array():
    """Test that empty arrays are handled gracefully."""
    x = np.array([])
    psi = mellin_cepstral_phase(x)
    assert math.isnan(psi)


def test_mellin_phase_handles_nan_input():
    """Test that NaN inputs are handled gracefully."""
    x = np.array([1.0, 2.0, np.nan, 3.0])
    psi = mellin_cepstral_phase(x)
    assert math.isnan(psi)


def test_rolling_mellin_handles_short_series():
    """Test rolling Mellin phase with series shorter than window."""
    series = pd.Series([1.0, 1.0, 1.0])
    res = rolling_mellin_cepstral_phase(series, window=8)
    assert len(res) == len(series)
    assert res.isna().all()


def test_mellin_cepstral_phase_with_debug():
    """Test debug output for Mellin cepstral phase."""
    rng = np.random.default_rng(789)
    x = rng.normal(size=64) + 2.0
    psi, debug = mellin_cepstral_phase(x, grid_n=128, return_debug=True)
    assert 0.0 <= psi < 1.0
    assert "psi_n_star" in debug
    assert "psi_c_real" in debug
    assert "psi_c_imag" in debug
    assert "psi_c_abs" in debug
    assert "psi_angle_rad" in debug
    assert "psi" in debug
    assert debug["psi"] == pytest.approx(psi)
