import numpy as np
import pandas as pd

from spot_bot.features import FeatureConfig, compute_features


def _synthetic_ohlcv(rows: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h")
    base = np.linspace(100.0, 120.0, rows)
    close = base + np.sin(np.linspace(0, np.pi, rows)) * 0.5
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        }
    )


def test_feature_pipeline_columns_and_ranges():
    cfg = FeatureConfig(rv_window=4, conc_window=8, psi_window=8, psi_mode="scale_phase")
    feats = compute_features(_synthetic_ohlcv(), cfg=cfg)

    for col in ["rv", "C", "psi", "C_int", "S"]:
        assert col in feats.columns
    psi_vals = feats["psi"].dropna()
    if not psi_vals.empty:
        assert ((psi_vals >= 0.0) & (psi_vals < 1.0)).all()

    valid_s = feats["S"].dropna()
    if not valid_s.empty:
        assert (valid_s >= 0.0).all() and (valid_s <= 1.0).all()

    for col in ["C", "C_int"]:
        valid = feats[col].dropna()
        if not valid.empty:
            assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_feature_pipeline_is_deterministic():
    cfg = FeatureConfig(rv_window=4, conc_window=8, psi_window=8, psi_mode="scale_phase")
    df = _synthetic_ohlcv(rows=32)
    feats1 = compute_features(df, cfg=cfg)
    feats2 = compute_features(df.copy(), cfg=cfg)

    pd.testing.assert_frame_equal(feats1[["rv", "C", "psi", "C_int", "S"]], feats2[["rv", "C", "psi", "C_int", "S"]])
