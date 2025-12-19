"""
Smoke test for real data evaluation script.

Tests that the evaluation runner can be imported and executed successfully
in fast mode with the committed data sample.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_real_data_eval_smoke():
    """
    Smoke test: Run evaluation script in fast mode and verify outputs.
    
    This test ensures:
    1. The data file exists
    2. The evaluation script runs without errors
    3. The report file is generated
    4. Key metrics are computed (not NaN)
    """
    # Check if data file exists
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "BTCUSDT_1H_real.csv.gz"
    
    if not data_path.exists():
        pytest.skip(f"Data file not found: {data_path}")
    
    # Verify data can be loaded
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    assert len(df) > 0, "Data file is empty"
    assert "close" in df.columns, "Missing 'close' column in data"
    
    # Run evaluation script in fast mode
    report_path = repo_root / "reports" / "DUAL_STREAM_REAL_DATA_REPORT.md"
    
    # Clean up old report if exists
    if report_path.exists():
        report_path.unlink()
    
    # Run the evaluation script
    cmd = [
        sys.executable,
        "-m",
        "theta_bot_averaging.eval.evaluate_dual_stream_real",
        "--fast",
        "--data-path",
        str(data_path),
        "--output",
        str(report_path),
    ]
    
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes max
    )
    
    # Check script executed successfully
    assert result.returncode == 0, f"Script failed with:\n{result.stderr}"
    
    # Verify report was generated
    assert report_path.exists(), "Report file was not generated"
    
    # Read and verify report content
    report_text = report_path.read_text()
    
    # Check for key sections
    assert "# Dual-Stream Real Data Evaluation Report" in report_text
    assert "Dataset Summary" in report_text
    assert "Configuration" in report_text
    assert "Results Comparison" in report_text
    assert "Predictive Metrics" in report_text
    assert "Trading Metrics" in report_text
    assert "Conclusion" in report_text
    
    # Verify that metrics tables are present
    assert "Baseline" in report_text
    assert "Dual-Stream" in report_text
    assert "Correlation" in report_text
    assert "Sharpe Ratio" in report_text
    assert "Total Return" in report_text
    
    # Verify numeric values are present (not just NaN)
    # The report should have percentage signs and actual numbers
    assert "%" in report_text, "No percentage values found in report"
    
    # Check for new required fields in report
    assert "min_close" in report_text, "Report missing min_close field"
    assert "max_close" in report_text, "Report missing max_close field"
    assert "is_realistic" in report_text, "Report missing is_realistic field"
    assert "predicted_return_std" in report_text, "Report missing predicted_return_std for models"
    assert "DATA SOURCE: REAL MARKET SAMPLE (validated)" in report_text, "Report missing data source validation statement"
    
    # Check that output indicates success
    assert "EVALUATION COMPLETE" in result.stdout or "Report written" in result.stdout
    
    print(f"✓ Smoke test passed")
    print(f"  - Data file: {data_path}")
    print(f"  - Report file: {report_path}")
    print(f"  - Report size: {len(report_text)} characters")


def test_evaluation_script_importable():
    """Test that the evaluation module can be imported."""
    try:
        from theta_bot_averaging.eval import evaluate_dual_stream_real
        assert hasattr(evaluate_dual_stream_real, "main")
    except ImportError as e:
        pytest.fail(f"Failed to import evaluation module: {e}")


def test_data_file_format():
    """Test that the data file has the correct format."""
    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "BTCUSDT_1H_real.csv.gz"
    
    if not data_path.exists():
        pytest.skip(f"Data file not found: {data_path}")
    
    # Load and verify data structure
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Check required columns
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check index is datetime
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    
    # Check data is not all NaN
    assert not df.isna().all().all(), "Data contains only NaN values"
    
    # Check OHLC consistency (basic sanity check)
    assert (df["high"] >= df["low"]).all(), "High must be >= Low"
    # High should be >= max(open, close), Low should be <= min(open, close)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), "High must be >= max(open, close)"
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), "Low must be <= min(open, close)"
    
    print(f"✓ Data file format validated")
    print(f"  - Shape: {df.shape}")
    print(f"  - Date range: {df.index[0]} to {df.index[-1]}")
