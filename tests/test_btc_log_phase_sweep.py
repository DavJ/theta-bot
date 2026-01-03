import math
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from btc_log_phase_sweep import (
    build_candidate_series,
    build_targets,
    evaluate_candidate,
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
