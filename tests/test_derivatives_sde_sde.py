import numpy as np
import pandas as pd

from theta_bot_averaging.derivatives_sde.state import build_state_from_frames
from theta_bot_averaging.derivatives_sde.sde_decompose import gate_lambda
from theta_bot_averaging.derivatives_sde.sigma_model import compute_sigma
from theta_bot_averaging.derivatives_sde.eval import evaluate_bias


def _synthetic_frames():
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    spot = pd.DataFrame({"close": np.linspace(100, 105, len(idx))}, index=idx)
    funding = pd.DataFrame({"fundingRate": np.linspace(0.0, 0.5, len(idx))}, index=idx)
    oi = pd.DataFrame({"sumOpenInterest": np.linspace(10, 20, len(idx))}, index=idx)
    basis = pd.DataFrame({"basis": np.linspace(-0.1, 0.2, len(idx))}, index=idx)
    return spot, funding, oi, basis


def test_state_alignment_and_trailing_zscore():
    spot, funding, oi, basis = _synthetic_frames()
    # remove first funding point to force intersection drop
    funding = funding.iloc[1:]
    state_df = build_state_from_frames(spot, funding, oi, basis, z_window=2, align="inner")
    assert state_df.index[0] == funding.index[0]

    # Trailing z-score should depend only on current/past observations
    series = funding["fundingRate"].reindex(state_df.index)
    expected = (series.iloc[1] - np.mean(series.iloc[:2])) / np.std(series.iloc[:2], ddof=1)
    assert np.isclose(state_df["z_funding"].iloc[1], expected, atol=1e-9)


def test_sigma_positive_floor():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    returns = pd.Series(0.0, index=idx)
    sigma = compute_sigma(returns, window=3)
    assert (sigma > 0).all()


def test_gate_lambda_quantile_and_threshold():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    lam = pd.Series([0.5, 1.0, 1.5, 2.0, 0.1], index=idx)
    active_q = gate_lambda(lam, tau_quantile=0.6)
    assert active_q.sum() > 0
    active_tau = gate_lambda(lam, tau=1.0)
    assert active_tau.equals(lam > 1.0)


def test_evaluation_detects_bias_on_synthetic_data():
    idx = pd.date_range("2024-01-01", periods=36, freq="h", tz="UTC")
    mu = pd.Series(np.concatenate([np.ones(18) * 0.1, np.ones(18) * -0.1]), index=idx)
    r = mu * 0.05  # realized returns follow mu direction
    df = pd.DataFrame({"mu": mu, "r": r, "Lambda": mu.abs() * 10, "active": True}, index=idx)

    res = evaluate_bias(df, horizons=[1, 3])
    assert res[1]["active"]["sign_agreement"] > 0.8
    assert res[3]["active"]["sign_agreement"] > 0.8
