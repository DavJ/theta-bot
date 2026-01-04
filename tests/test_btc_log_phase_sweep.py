import math
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from btc_log_phase_sweep import (
    add_residualized_internal_phase,
    build_candidate_series,
    build_targets,
    compute_features,
    compute_internal_phase,
    evaluate_candidate,
    _cepstral_phase,
    rolling_torus_concentration,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "close": [100.0, 105.0, 102.0, 110.0],
            "high": [101.0, 106.0, 103.0, 111.0],
            "low": [99.0, 104.0, 101.0, 108.0],
            "volume": [10.0, 11.0, 12.0, 13.0],
        }
    )


def test_candidate_builders_positive_and_handle_windows(sample_df):
    args = Namespace(rv_window=2, atr_window=2, volume_roll=2, ema_window=2)

    abslogret = build_candidate_series(sample_df, "abslogret", args)
    assert (abslogret.dropna() >= 0).all()

    rv = build_candidate_series(sample_df, "rv", args)
    assert pd.isna(rv.iloc[0])
    assert rv.iloc[2] >= 0

    atr = build_candidate_series(sample_df, "atr", args)
    assert atr.iloc[1] >= 0

    volume_roll = build_candidate_series(sample_df, "volume", args)
    assert volume_roll.iloc[1] == pytest.approx(21.0)

    distema = build_candidate_series(sample_df, "distema", args)
    assert (distema.dropna() >= 0).all()


def test_targets_future_alignment_no_lookahead():
    args = Namespace(target_window=2, horizon=2)
    df = pd.DataFrame({"close": [1.0, 2.0, 4.0, 8.0]})
    targets = build_targets(df, args)
    assert targets["y_vol"].iloc[0] == pytest.approx(0.0)
    assert targets["y_vol"].iloc[1] == pytest.approx(0.0)
    assert targets["y_absret"].iloc[0] == pytest.approx(math.log(4.0))
    assert pd.isna(targets["y_absret"].iloc[-2])


def test_evaluate_candidate_returns_metrics_and_buckets():
    phi = np.linspace(0, 1, 10, endpoint=False)
    cos_phi = np.cos(2 * np.pi * phi)
    sin_phi = np.sin(2 * np.pi * phi)
    concentration = np.linspace(0.1, 0.9, 10)
    features = pd.DataFrame(
        {"phi": phi, "cos_phi": cos_phi, "sin_phi": sin_phi, "concentration": concentration}
    )
    targets = pd.DataFrame(
        {"y_vol": np.linspace(1.0, 2.0, 10), "y_absret": np.linspace(0.5, 1.5, 10)}
    )
    res = evaluate_candidate(features, targets)
    assert "ic_conc_y_vol" in res and res["ic_conc_y_vol"] is not None
    assert "bucket_counts" in res and sum(res["bucket_counts"].values()) == len(features)
    assert res["bucket_ratio"] >= 1.0


def test_evaluate_candidate_includes_ensemble_score():
    phi = np.linspace(0, 1, 12, endpoint=False)
    cos_phi = np.cos(2 * np.pi * phi)
    sin_phi = np.sin(2 * np.pi * phi)
    concentration = np.linspace(0.1, 0.9, 12)
    c_int = np.linspace(0.2, 1.0, 12)
    features = pd.DataFrame(
        {
            "phi": phi,
            "cos_phi": cos_phi,
            "sin_phi": sin_phi,
            "concentration": concentration,
            "c_int": c_int,
        }
    )
    targets = pd.DataFrame(
        {"y_vol": np.linspace(1.0, 3.0, 12), "y_absret": np.linspace(0.5, 1.5, 12)}
    )
    res = evaluate_candidate(features, targets)
    assert not math.isnan(res["ic_s_y_vol"])
    assert res["s_bucket_ratio"] >= 1.0
    assert res["s_bucket_counts"]


def test_rolling_torus_concentration_range_and_basic_behavior():
    # Perfectly aligned phases -> high concentration
    n = 20
    cos_s = np.ones(n)
    sin_s = np.zeros(n)
    cos_t = np.ones(n)
    sin_t = np.zeros(n)
    c = rolling_torus_concentration(cos_s, sin_s, cos_t, sin_t, window=10)
    assert np.nanmax(c) <= 1.0 + 1e-9
    assert np.nanmin(c[9:]) >= 0.99

    # Random time circle should reduce concentration on average
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    cos_t2 = np.cos(angles)
    sin_t2 = np.sin(angles)
    c2 = rolling_torus_concentration(cos_s, sin_s, cos_t2, sin_t2, window=10)
    assert np.nanmean(c2[9:]) < 0.99


def test_hilbert_rv_psi_within_unit_interval():
    idx = pd.date_range("2024-01-01", periods=12, freq="h")
    close = np.array([100.0, 102.0, 101.0, 105.0, 110.0, 108.0, 112.0, 115.0, 117.0, 120.0, 118.0, 121.0])
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.linspace(1.0, 2.0, len(idx)),
            "dt": idx,
        }
    )
    args = Namespace(
        base=10.0,
        conc_window=4,
        rv_window=3,
        psi_mode="hilbert_rv",
        psi_window=4,
    )
    features = compute_features(df["close"], df, args)
    psi_vals = features["psi"].dropna()
    assert not psi_vals.empty
    assert ((psi_vals >= 0.0) & (psi_vals < 1.0)).all()


def test_hilbert_rv_psi_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=12, freq="h")
    close = np.array([100.0, 102.0, 101.0, 105.0, 110.0, 108.0, 112.0, 115.0, 117.0, 120.0, 118.0, 121.0])
    df1 = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.linspace(1.0, 2.0, len(idx)),
            "dt": idx,
        }
    )
    df2 = df1.copy()
    df2.loc[df2.index[-1], "close"] *= 5.0  # perturb only the final point to ensure causality
    args = Namespace(
        base=10.0,
        conc_window=4,
        rv_window=3,
        psi_mode="hilbert_rv",
        psi_window=4,
    )
    features1 = compute_features(df1["close"], df1, args)
    features2 = compute_features(df2["close"], df2, args)
    compare_idx = len(idx) - 2
    v1 = features1["psi"].iloc[compare_idx]
    v2 = features2["psi"].iloc[compare_idx]
    assert math.isfinite(v1)
    assert v1 == pytest.approx(v2)


def test_c_int_computed_when_psi_enabled():
    idx = pd.date_range("2024-01-01", periods=16, freq="h")
    close = np.linspace(100.0, 120.0, len(idx))
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.linspace(1.0, 2.0, len(idx)),
            "dt": idx,
        }
    )
    args = Namespace(base=10.0, conc_window=4, rv_window=3, psi_mode="hilbert_rv", psi_window=4)
    psi = compute_internal_phase(df, args)
    features = compute_features(df["close"], df, args, psi_series=psi)
    assert "c_int" in features.columns
    c_int_vals = features["c_int"].dropna()
    assert not c_int_vals.empty
    assert ((c_int_vals >= 0.0) & (c_int_vals <= 1.0)).all()


def test_psi_residualization_uses_train_fit_for_c_int():
    n = 6
    conc_window = 2
    concentration = np.linspace(0.0, 0.5, n)
    psi = 0.3 + 0.2 * concentration
    phi = np.zeros(n)
    features = pd.DataFrame(
        {
            "phi": phi,
            "cos_phi": np.cos(2 * np.pi * phi),
            "sin_phi": np.sin(2 * np.pi * phi),
            "concentration": concentration,
            "psi": psi,
        }
    )
    add_residualized_internal_phase(features, conc_window=conc_window, train_end=3)
    assert "psi_perp" in features.columns
    assert np.allclose(features["psi_perp"].iloc[:3], 0.0, atol=1e-8)
    assert "c_int_resid" in features.columns
    assert math.isfinite(features["c_int_resid"].iloc[-1])


def test_different_psi_modes_produce_different_series():
    idx = pd.date_range("2024-01-01", periods=20, freq="h")
    close = np.sin(np.linspace(0, 4 * np.pi, len(idx))) * 5 + 100.0
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.linspace(1.0, 2.0, len(idx)),
            "dt": idx,
        }
    )
    args_hilbert = Namespace(
        base=10.0,
        conc_window=4,
        rv_window=3,
        psi_mode="hilbert_rv",
        psi_window=4,
    )
    args_cepstrum = Namespace(
        base=10.0,
        conc_window=4,
        rv_window=3,
        psi_mode="cepstrum",
        psi_window=4,
    )
    psi_hilbert = compute_internal_phase(df, args_hilbert)
    psi_cepstrum = compute_internal_phase(df, args_cepstrum)
    overlap_idx = psi_hilbert.dropna().index.intersection(psi_cepstrum.dropna().index)
    assert not overlap_idx.empty
    assert not np.allclose(psi_hilbert.loc[overlap_idx], psi_cepstrum.loc[overlap_idx])


def test_cepstrum_parameters_affect_output_and_stay_in_range():
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    close = np.sin(np.linspace(0, 6 * np.pi, len(idx))) * 3 + 100.0
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.linspace(1.0, 2.0, len(idx)),
            "dt": idx,
        }
    )
    base_kwargs = dict(base=10.0, conc_window=6, rv_window=4, psi_window=8)
    args_default = Namespace(psi_mode="cepstrum", **base_kwargs)
    args_tuned = Namespace(
        psi_mode="cepstrum",
        cepstrum_min_bin=3,
        cepstrum_max_frac=0.5,
        cepstrum_topk=3,
        **base_kwargs,
    )
    psi_default = compute_internal_phase(df, args_default)
    psi_tuned = compute_internal_phase(df, args_tuned)
    psi_tuned_vals = psi_tuned.dropna()
    assert not psi_tuned_vals.empty
    assert ((psi_tuned_vals >= 0.0) & (psi_tuned_vals < 1.0)).all()
    overlap = psi_default.dropna().index.intersection(psi_tuned_vals.index)
    assert not overlap.empty
    assert not np.allclose(psi_default.loc[overlap], psi_tuned.loc[overlap])


def test_cepstrum_logtime_stays_in_range():
    idx = pd.date_range("2024-02-01", periods=30, freq="h")
    close = np.linspace(90.0, 110.0, len(idx)) + 0.5 * np.sin(np.linspace(0, 4 * np.pi, len(idx)))
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "volume": np.linspace(1.0, 2.0, len(idx)),
            "dt": idx,
        }
    )
    args = Namespace(
        base=10.0,
        conc_window=6,
        rv_window=4,
        psi_window=8,
        psi_mode="cepstrum",
        cepstrum_domain="logtime",
    )
    psi_logtime = compute_internal_phase(df, args)
    psi_vals = psi_logtime.dropna()
    assert not psi_vals.empty
    assert ((psi_vals >= 0.0) & (psi_vals < 1.0)).all()


def test_cepstrum_logtime_differs_from_linear_on_chirp():
    window = 32
    t = np.linspace(0.0, 1.0, 80)
    chirp = signal.chirp(t, f0=0.5, f1=6.0, t1=1.0, method="logarithmic")
    series = pd.Series(chirp)
    psi_linear = _cepstral_phase(series, window=window, domain="linear")
    psi_logtime = _cepstral_phase(series, window=window, domain="logtime")
    overlap = psi_linear.dropna().index.intersection(psi_logtime.dropna().index)
    assert not overlap.empty
    assert not np.allclose(psi_linear.loc[overlap], psi_logtime.loc[overlap])
