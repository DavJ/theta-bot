import numpy as np
import pandas as pd

from theta_features.cepstrum import rolling_cepstral_phase

WINDOW = 256
PRIMARY_FREQ_BIN = 5
SECONDARY_FREQ_BIN = 11
SECONDARY_AMPLITUDE = 0.5
PHASE_WALK_SCALE = 0.1
MIN_UNIQUE_VALUES = 20


def test_psi_not_degenerate_and_in_range():
    rng = np.random.default_rng(123)
    n = 800
    t = np.arange(n)
    omega1 = 2 * np.pi * PRIMARY_FREQ_BIN / WINDOW
    omega2 = 2 * np.pi * SECONDARY_FREQ_BIN / WINDOW
    phase_walk = np.cumsum(rng.normal(scale=PHASE_WALK_SCALE, size=n))
    signal = np.sin(omega1 * t + phase_walk) + SECONDARY_AMPLITUDE * np.sin(
        omega2 * t + rng.uniform(0, 2 * np.pi)
    )

    psi = rolling_cepstral_phase(pd.Series(signal), window=WINDOW)
    vals = psi.dropna().to_numpy(dtype=float)

    assert vals.size > 0
    assert np.all(vals >= 0.0)
    assert np.all(vals < 1.0)
    assert not np.any(np.isclose(vals, 1.0))

    rounded_unique = np.unique(np.round(vals, 2))
    assert rounded_unique.size > MIN_UNIQUE_VALUES
