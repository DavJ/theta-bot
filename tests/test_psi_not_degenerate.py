import numpy as np
import pandas as pd

from theta_features.cepstrum import rolling_cepstral_phase


def test_psi_not_degenerate_and_in_range():
    rng = np.random.default_rng(123)
    window = 256
    n = 800
    t = np.arange(n)
    omega1 = 2 * np.pi * 5 / window
    omega2 = 2 * np.pi * 11 / window
    phase_walk = np.cumsum(rng.normal(scale=0.1, size=n))
    signal = np.sin(omega1 * t + phase_walk) + 0.5 * np.sin(omega2 * t + rng.uniform(0, 2 * np.pi))

    psi = rolling_cepstral_phase(pd.Series(signal), window=window)
    vals = psi.dropna().to_numpy(dtype=float)

    assert vals.size > 0
    assert np.all(vals >= 0.0)
    assert np.all(vals < 1.0)
    assert not np.any(np.isclose(vals, 1.0))

    rounded_unique = np.unique(np.round(vals, 2))
    assert rounded_unique.size > 20
