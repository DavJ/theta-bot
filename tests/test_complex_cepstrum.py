import numpy as np
import pandas as pd

from theta_features.cepstrum import complex_cepstrum, rolling_complex_cepstral_phase


def _phase_modulated_signal(length: int) -> np.ndarray:
    t = np.arange(length, dtype=float)
    base = 2 * np.pi * t / length
    carrier = np.sin(5 * base)
    modulated = 0.5 * np.sin(2 * base + 0.3 * np.sin(base))
    return carrier + modulated


def test_complex_cepstrum_has_nontrivial_imaginary_part():
    sig = _phase_modulated_signal(128)
    c = complex_cepstrum(sig)
    assert np.max(np.abs(np.imag(c))) > 1e-6


def test_rolling_complex_cepstral_phase_varies_and_in_range():
    sig = _phase_modulated_signal(256)
    series = pd.Series(sig)
    psi, debug = rolling_complex_cepstral_phase(
        series, window=80, min_bin=4, max_frac=0.3, domain="linear", return_debug=True
    )
    vals = psi.dropna()
    assert not vals.empty
    assert ((vals >= 0.0) & (vals < 1.0)).all()
    assert vals.round(3).nunique() >= 10
    assert debug["psi_c_imag"].abs().max() > 1e-6
