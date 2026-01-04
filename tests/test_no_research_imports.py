import sys

import pandas as pd

from spot_bot.features.feature_pipeline import FeatureConfig, compute_features


def test_feature_pipeline_avoids_research_imports():
    sys.modules.pop("btc_log_phase_sweep", None)

    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [0.9, 1.0, 1.1, 1.2],
            "close": [1.05, 1.15, 1.25, 1.35],
            "volume": [10, 11, 12, 13],
        },
        index=idx,
    )
    cfg = FeatureConfig(rv_window=2, conc_window=2, psi_window=2, cepstrum_min_bin=1, cepstrum_max_frac=0.5)
    features = compute_features(df, cfg)
    assert not features.empty
    assert "btc_log_phase_sweep" not in sys.modules
