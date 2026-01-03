"""
Smoke test for phase-collapse event prediction evaluation.

Tests that the evaluation script runs without errors in fast mode.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_eval_phase_collapse_smoke():
    """
    Run evaluate_phase_collapse in --fast mode and verify it completes successfully.
    """
    # Get repo root
    repo_root = Path(__file__).resolve().parent.parent
    
    # Check if data file exists
    data_file = repo_root / "data" / "BTCUSDT_1H_real.csv.gz"
    if not data_file.exists():
        pytest.skip(f"Data file not found: {data_file}")
    
    # Run evaluation in fast mode
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "theta_bot_averaging.eval.evaluate_phase_collapse",
            "--fast",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    
    # Check exit code
    assert result.returncode == 0, (
        f"Evaluation failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    
    # Check that report was generated
    report_path = repo_root / "reports" / "PHASE_COLLAPSE_EVAL_REPORT.md"
    assert report_path.exists(), f"Report file not created: {report_path}"
    
    # Verify report contains expected content
    report_text = report_path.read_text()
    
    assert "Phase-Collapse Event Prediction Evaluation Report" in report_text, \
        "Report missing title"
    assert "ROC-AUC" in report_text, "Report missing ROC-AUC metric"
    assert "PR-AUC" in report_text, "Report missing PR-AUC metric"
    assert "Gating Sanity Check" in report_text, "Report missing gating sanity check"
    assert "VOL_BURST" in report_text, "Report missing event type"
    
    # Verify that both baseline and dual-stream results are present
    assert "Baseline" in report_text, "Report missing baseline results"
    assert "Dual-Stream" in report_text, "Report missing dual-stream results"
    
    print("✓ Smoke test passed: evaluation completed successfully")
    print(f"✓ Report generated: {report_path}")


def test_eval_phase_collapse_import():
    """
    Test that the module can be imported without errors.
    """
    try:
        from theta_bot_averaging.eval import evaluate_phase_collapse
        assert hasattr(evaluate_phase_collapse, 'main'), "Module missing main function"
    except ImportError as e:
        pytest.fail(f"Failed to import evaluate_phase_collapse: {e}")


def test_targets_import():
    """
    Test that the targets module can be imported.
    """
    try:
        from theta_bot_averaging.targets import make_vol_burst_labels
        assert callable(make_vol_burst_labels), "make_vol_burst_labels should be callable"
    except ImportError as e:
        pytest.fail(f"Failed to import targets module: {e}")
