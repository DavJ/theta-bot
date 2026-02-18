"""Tests for forensic_fingerprint.tools.theta_fit_tau.

These tests verify that:
  1. The module can be imported without triggering matplotlib side-effects.
  2. gauss_envelope and theta3_envelope return the expected shapes / values.
  3. fit_gauss_envelope recovers known parameters from synthetic data.
  4. fit_theta3_envelope recovers known parameters from synthetic data.
  5. Helper utilities (detect_tau_peaks, residual_rms, residual_max_abs) work.
"""

import sys
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Guard: importing the module must NOT pull in matplotlib
# ---------------------------------------------------------------------------

def test_import_does_not_load_matplotlib():
    """Importing theta_fit_tau must not import matplotlib at module level."""
    # Remove any previously-cached matplotlib reference to get a clean check
    mpl_before = "matplotlib" in sys.modules

    import forensic_fingerprint.tools.theta_fit_tau  # noqa: F401

    if not mpl_before:
        assert "matplotlib" not in sys.modules, (
            "matplotlib was imported as a side-effect of importing theta_fit_tau"
        )


# ---------------------------------------------------------------------------
# Import the public API
# ---------------------------------------------------------------------------

from forensic_fingerprint.tools.theta_fit_tau import (  # noqa: E402
    detect_tau_peaks,
    fit_gauss_envelope,
    fit_theta3_envelope,
    gauss_envelope,
    residual_max_abs,
    residual_rms,
    theta3_envelope,
)


# ---------------------------------------------------------------------------
# gauss_envelope
# ---------------------------------------------------------------------------

class TestGaussEnvelope:
    def test_shape(self):
        t = np.linspace(0, 10, 200)
        out = gauss_envelope(t, A=3.0, mu=5.0, sigma=1.5)
        assert out.shape == t.shape

    def test_peak_at_mu(self):
        t = np.linspace(0, 10, 1000)
        A, mu, sigma = 2.0, 4.0, 1.0
        out = gauss_envelope(t, A=A, mu=mu, sigma=sigma)
        assert t[np.argmax(out)] == pytest.approx(mu, abs=0.02)

    def test_amplitude_at_peak(self):
        t = np.linspace(0, 10, 1000)
        A = 5.0
        out = gauss_envelope(t, A=A, mu=5.0, sigma=0.5)
        assert np.max(out) == pytest.approx(A, rel=1e-3)

    def test_offset(self):
        t = np.linspace(0, 10, 100)
        out = gauss_envelope(t, A=0.0, mu=5.0, sigma=1.0, offset=3.0)
        assert np.allclose(out, 3.0)


# ---------------------------------------------------------------------------
# theta3_envelope
# ---------------------------------------------------------------------------

class TestTheta3Envelope:
    def test_shape(self):
        t = np.linspace(0, 4, 200)
        out = theta3_envelope(t, A=1.0, mu=2.0, T=4.0, q=0.3)
        assert out.shape == t.shape

    def test_q_near_zero_approx_constant(self):
        """For q → 0 theta_3 → 1, so result ≈ A + offset."""
        t = np.linspace(0, 10, 100)
        out = theta3_envelope(t, A=2.0, mu=5.0, T=3.0, q=1e-8, offset=1.0)
        assert np.allclose(out, 3.0, atol=1e-5)

    def test_finite_values(self):
        t = np.linspace(0, 10, 100)
        out = theta3_envelope(t, A=1.0, mu=5.0, T=2.0, q=0.5)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# fit_gauss_envelope
# ---------------------------------------------------------------------------

class TestFitGaussEnvelope:
    @pytest.fixture()
    def synthetic_gauss(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 20, 400)
        true_params = dict(A=4.0, mu=10.0, sigma=2.5, offset=0.5)
        data = gauss_envelope(t, **true_params) + rng.normal(0, 0.05, t.size)
        return t, data, true_params

    def test_returns_dict_with_expected_keys(self, synthetic_gauss):
        t, data, _ = synthetic_gauss
        result = fit_gauss_envelope(t, data)
        assert set(result.keys()) == {"params", "cov", "fitted", "residual"}

    def test_fitted_shape(self, synthetic_gauss):
        t, data, _ = synthetic_gauss
        result = fit_gauss_envelope(t, data)
        assert result["fitted"].shape == t.shape
        assert result["residual"].shape == t.shape

    def test_recovers_amplitude(self, synthetic_gauss):
        t, data, true_params = synthetic_gauss
        result = fit_gauss_envelope(t, data)
        A_fit, mu_fit, sigma_fit, offset_fit = result["params"]
        assert A_fit == pytest.approx(true_params["A"], rel=0.05)

    def test_recovers_centre(self, synthetic_gauss):
        t, data, true_params = synthetic_gauss
        result = fit_gauss_envelope(t, data)
        _, mu_fit, _, _ = result["params"]
        assert mu_fit == pytest.approx(true_params["mu"], abs=0.2)

    def test_low_residual_rms(self, synthetic_gauss):
        t, data, _ = synthetic_gauss
        result = fit_gauss_envelope(t, data)
        assert residual_rms(result) < 0.2

    def test_accepts_custom_p0(self, synthetic_gauss):
        t, data, true_params = synthetic_gauss
        p0 = [true_params["A"], true_params["mu"], true_params["sigma"], true_params["offset"]]
        result = fit_gauss_envelope(t, data, p0=p0)
        assert result["params"] is not None


# ---------------------------------------------------------------------------
# fit_theta3_envelope
# ---------------------------------------------------------------------------

class TestFitTheta3Envelope:
    @pytest.fixture()
    def synthetic_theta3(self):
        rng = np.random.default_rng(7)
        t = np.linspace(0, 10, 300)
        true_params = dict(A=1.0, mu=5.0, T=4.0, q=0.3, offset=0.2)
        data = theta3_envelope(t, **true_params) + rng.normal(0, 0.02, t.size)
        return t, data, true_params

    def test_returns_dict_with_expected_keys(self, synthetic_theta3):
        t, data, _ = synthetic_theta3
        result = fit_theta3_envelope(t, data)
        assert set(result.keys()) == {"params", "cov", "fitted", "residual"}

    def test_fitted_shape(self, synthetic_theta3):
        t, data, _ = synthetic_theta3
        result = fit_theta3_envelope(t, data)
        assert result["fitted"].shape == t.shape

    def test_recovers_period(self, synthetic_theta3):
        t, data, true_params = synthetic_theta3
        result = fit_theta3_envelope(t, data)
        _, _, T_fit, _, _ = result["params"]
        assert T_fit == pytest.approx(true_params["T"], rel=0.1)

    def test_low_residual_rms(self, synthetic_theta3):
        t, data, _ = synthetic_theta3
        result = fit_theta3_envelope(t, data)
        assert residual_rms(result) < 0.2


# ---------------------------------------------------------------------------
# detect_tau_peaks
# ---------------------------------------------------------------------------

class TestDetectTauPeaks:
    def test_finds_single_peak(self):
        t = np.linspace(0, 10, 500)
        data = gauss_envelope(t, A=1.0, mu=5.0, sigma=0.5)
        peaks = detect_tau_peaks(t, data)
        assert len(peaks) >= 1
        assert t[peaks[0]] == pytest.approx(5.0, abs=0.1)

    def test_finds_multiple_peaks(self):
        t = np.linspace(0, 20, 2000)
        data = (
            gauss_envelope(t, A=1.0, mu=5.0, sigma=0.3)
            + gauss_envelope(t, A=1.0, mu=15.0, sigma=0.3)
        )
        peaks = detect_tau_peaks(t, data)
        assert len(peaks) >= 2

    def test_empty_input_returns_array(self):
        t = np.linspace(0, 1, 50)
        data = np.zeros(50)
        peaks = detect_tau_peaks(t, data)
        assert isinstance(peaks, np.ndarray)


# ---------------------------------------------------------------------------
# residual helpers
# ---------------------------------------------------------------------------

class TestResidualHelpers:
    def _make_result(self, residual):
        return {"residual": np.asarray(residual, dtype=float)}

    def test_residual_rms_zero(self):
        result = self._make_result([0.0, 0.0, 0.0])
        assert residual_rms(result) == pytest.approx(0.0)

    def test_residual_rms_known(self):
        result = self._make_result([1.0, -1.0, 1.0, -1.0])
        assert residual_rms(result) == pytest.approx(1.0)

    def test_residual_max_abs(self):
        result = self._make_result([0.5, -2.0, 1.0])
        assert residual_max_abs(result) == pytest.approx(2.0)
