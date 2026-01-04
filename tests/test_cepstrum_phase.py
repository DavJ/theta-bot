import math

import numpy as np
import pandas as pd
import pytest

from theta_features.cepstrum import cepstral_phase, rolling_cepstral_phase


def test_constant_signal_returns_zero_phase():
    x = np.ones(16, dtype=float)
    psi = cepstral_phase(x, domain="linear", min_bin=1, max_frac=0.5)
    assert 0.0 <= psi < 1.0
    assert psi == pytest.approx(0.0)


def test_cepstral_phase_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=64) + 2.0
    psi1 = cepstral_phase(x, domain="linear", min_bin=2, max_frac=0.4, topk=3)
    psi2 = cepstral_phase(x, domain="linear", min_bin=2, max_frac=0.4, topk=3)
    assert not math.isnan(psi1)
    assert psi1 == pytest.approx(psi2)
    assert 0.0 <= psi1 < 1.0


def test_rolling_cepstral_phase_handles_short_and_constant_series():
    series = pd.Series([1.0, 1.0, 1.0])
    res = rolling_cepstral_phase(series, window=4)
    assert len(res) == len(series)
    assert res.isna().all()


def test_linear_and_logtime_outputs_in_range():
    idx = pd.date_range("2024-01-01", periods=12, freq="h")
    series = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx)
    res_linear = rolling_cepstral_phase(series, window=6, domain="linear")
    res_logtime = rolling_cepstral_phase(series, window=6, domain="logtime")

    for res in (res_linear, res_logtime):
        vals = res.dropna()
        assert not vals.empty
        assert ((vals >= 0.0) & (vals < 1.0)).all()


def test_noisy_input_stays_in_range():
    rng = np.random.default_rng(42)
    noisy = pd.Series(rng.normal(loc=1.0, scale=0.5, size=40))
    res = rolling_cepstral_phase(noisy, window=8, domain="linear", topk=2)
    vals = res.dropna()
    assert not vals.empty
    assert ((vals >= 0.0) & (vals < 1.0)).all()
