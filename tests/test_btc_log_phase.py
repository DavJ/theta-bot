import numpy as np
import pandas as pd
import pytest

from btc_log_phase import (
    circ_dist,
    frac,
    log_phase,
    phase_embedding,
    risk_filter_backtest,
    rolling_phase_concentration,
)


def test_frac_handles_negative_values():
    arr = np.array([1.2, -0.3, 2.0])
    res = frac(arr)
    assert np.allclose(res, [0.2, 0.7, 0.0])


def test_log_phase_and_embedding_return_expected_shapes():
    returns = np.array([1.0, 3.0])
    phi = log_phase(returns, base=10.0)
    assert phi.shape == (2,)
    assert phi[0] == pytest.approx(0.0)
    cos_phi, sin_phi = phase_embedding(phi)
    assert cos_phi.shape == (2,)
    assert sin_phi.shape == (2,)
    assert cos_phi[0] == pytest.approx(1.0)
    assert sin_phi[0] == pytest.approx(0.0)


def test_circ_dist_wraps_through_zero():
    assert circ_dist(0.95, 0.05) == pytest.approx(0.10)


def test_rolling_phase_concentration_basic_behavior():
    phi = np.array([0.0, 0.0, 0.0, 0.5])
    conc = rolling_phase_concentration(phi, window=3)
    assert np.isnan(conc[0]) and np.isnan(conc[1])
    assert conc[2] == pytest.approx(1.0)
    vectors = np.array([1, 1, -1], dtype=complex)
    expected_cluster = abs(vectors.mean())  # |mean(e^{i*2πφ})| over last 3 points
    assert conc[3] == pytest.approx(expected_cluster)


def test_risk_filter_backtest_weights_and_equity():
    df = pd.DataFrame(
        {
            "ret": [1.0, 1.01, 0.99, 1.02],
            "concentration": [0.1, 0.3, 0.1, 0.25],
        }
    )
    eq_bh, eq_filtered, summary = risk_filter_backtest(df, thr=0.2)
    assert eq_bh[-1] == pytest.approx(1.019898)
    assert eq_filtered[-1] == pytest.approx(0.99)
    assert summary["time_in_market"] == pytest.approx(0.5)
