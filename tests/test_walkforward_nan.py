import pandas as pd
import yaml

from theta_bot_averaging.validation import run_walkforward


def test_walkforward_handles_feature_nans(tmp_path):
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=150, freq="h")
    t = np.linspace(0, 6 * np.pi, len(idx))
    prices = 100 + np.sin(t) + 0.5 * np.sin(0.5 * t)
    volume = 100 + np.abs(np.cos(t)) * 10
    df = pd.DataFrame(
        {
            "open": prices + 0.1,
            "high": prices + 0.2,
            "low": prices - 0.1,
            "close": prices,
            "volume": volume,
        },
        index=idx,
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path)

    cfg_path = tmp_path / "config.yaml"
    cfg = {
        "data_path": str(data_path),
        "horizon": 1,
        "threshold_bps": 10.0,
        "model_type": "logit",
        "fee_rate": 0.0,
        "slippage_bps": 0.0,
        "spread_bps": 0.0,
        "n_splits": 3,
        "purge": 0,
        "embargo": 0,
        "output_dir": str(tmp_path / "runs"),
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    res = run_walkforward(str(cfg_path))

    assert "metrics" in res
    assert (tmp_path / "runs").exists()
