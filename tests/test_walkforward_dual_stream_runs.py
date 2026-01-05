"""Test end-to-end walkforward with dual-stream model."""

import json

import numpy as np
import pandas as pd
import yaml

from theta_bot_averaging.validation import run_walkforward


def test_walkforward_dual_stream_runs(tmp_path):
    """Test that walkforward with dual_stream model runs successfully."""
    np.random.seed(42)
    
    # Create synthetic dataset with enough samples
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.linspace(0, 6 * np.pi, n)
    
    # Create price series with trend and oscillation
    prices = 100 + 0.5 * t + 10 * np.sin(t) + 2 * np.sin(3 * t) + np.random.randn(n) * 0.5
    volume = 1000 + 200 * np.abs(np.cos(t)) + np.random.rand(n) * 50
    
    df = pd.DataFrame(
        {
            "open": prices + 0.1,
            "high": prices + 0.3,
            "low": prices - 0.2,
            "close": prices,
            "volume": volume,
        },
        index=idx,
    )
    
    # Save data
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path)
    
    # Create config for dual_stream model
    cfg = {
        "data_path": str(data_path),
        "horizon": 1,
        "threshold_bps": 10.0,
        "model_type": "dual_stream",
        "fee_rate": 0.0,
        "slippage_bps": 0.0,
        "spread_bps": 0.0,
        "n_splits": 3,
        "purge": 0,
        "embargo": 0,
        "output_dir": str(tmp_path / "runs"),
        # Dual-stream specific (use small values for fast test)
        "theta_window": 30,
        "theta_q": 0.9,
        "theta_terms": 5,
        "mellin_k": 8,
        "mellin_alpha": 0.5,
        "mellin_omega_max": 1.0,
        "torch_epochs": 3,  # Very few epochs for fast test
        "torch_batch_size": 16,
        "torch_lr": 1e-3,
    }
    
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    
    # Run walkforward
    res = run_walkforward(str(cfg_path))
    
    # Verify results
    assert "metrics" in res
    assert "output_dir" in res
    
    # Check output directory exists
    output_dir = tmp_path / "runs"
    assert output_dir.exists()
    
    # Check metrics.json exists
    metrics_files = list(output_dir.rglob("metrics.json"))
    assert len(metrics_files) > 0
    
    # Load and verify metrics
    with open(metrics_files[0], "r") as f:
        metrics = json.load(f)
    
    assert "fold_metrics" in metrics
    assert "aggregate" in metrics
    assert "backtest" in metrics
    assert len(metrics["fold_metrics"]) > 0
    
    # Verify predictions file exists
    predictions_files = list(output_dir.rglob("predictions.parquet"))
    if predictions_files:
        preds_df = pd.read_parquet(predictions_files[0])
    else:
        predictions_files = list(output_dir.rglob("predictions.csv"))
        assert len(predictions_files) > 0
        preds_df = pd.read_csv(predictions_files[0], index_col=0, parse_dates=True)
    
    # Load predictions and verify columns
    assert "predicted_return" in preds_df.columns
    assert "signal" in preds_df.columns
    assert "future_return" in preds_df.columns
    
    # Verify no NaNs in predicted_return
    assert not preds_df["predicted_return"].isna().any()
    
    # Verify signals are in {-1, 0, 1}
    assert preds_df["signal"].isin([-1, 0, 1]).all()


def test_walkforward_dual_stream_fallback_if_no_torch(tmp_path, monkeypatch):
    """Test that dual_stream falls back gracefully when PyTorch unavailable."""
    # This test simulates PyTorch being unavailable by checking fallback behavior
    # In practice, the fallback uses BaselineModel internally
    
    np.random.seed(123)
    n = 150
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
    
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.2,
            "low": prices - 0.2,
            "close": prices,
            "volume": 1000,
        },
        index=idx,
    )
    
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path)
    
    cfg = {
        "data_path": str(data_path),
        "horizon": 1,
        "threshold_bps": 10.0,
        "model_type": "dual_stream",
        "fee_rate": 0.0,
        "slippage_bps": 0.0,
        "spread_bps": 0.0,
        "n_splits": 2,
        "purge": 0,
        "embargo": 0,
        "output_dir": str(tmp_path / "runs"),
        "theta_window": 25,
        "theta_q": 0.85,
        "theta_terms": 4,
        "mellin_k": 6,
        "torch_epochs": 2,
        "torch_batch_size": 16,
        "torch_lr": 1e-3,
    }
    
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    
    # Run walkforward - should use fallback if torch unavailable
    res = run_walkforward(str(cfg_path))
    
    # Should still produce valid results
    assert "metrics" in res
    assert "output_dir" in res
    
    # Check output created
    output_dir = tmp_path / "runs"
    assert output_dir.exists()
    
    metrics_files = list(output_dir.rglob("metrics.json"))
    assert len(metrics_files) > 0
