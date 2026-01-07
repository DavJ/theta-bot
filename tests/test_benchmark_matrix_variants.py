import pytest

from bench.benchmark_matrix import _build_run_plan


def test_none_mode_runs_baseline_only():
    plan = _build_run_plan(["BTC/USDT"], ["none", "scale_phase"], ["C", "S"])
    assert ("BTC/USDT", "S", "none") not in plan
    assert ("BTC/USDT", "C", "none") in plan
    assert ("BTC/USDT", "S", "scale_phase") in plan
    assert ("BTC/USDT", "C", "scale_phase") in plan


def test_invalid_modes_rejected():
    with pytest.raises(ValueError):
        _build_run_plan(["BTC/USDT"], ["mellin_cepstrum"], ["C"])


def test_missing_valid_methods_rejected():
    with pytest.raises(ValueError):
        _build_run_plan(["BTC/USDT"], ["none"], ["S"])


def test_kalman_mr_dual_in_plan():
    """Test that KALMAN_MR_DUAL method is properly included in run plan."""
    plan = _build_run_plan(["BTC/USDT"], ["scale_phase"], ["KALMAN_MR_DUAL"])
    assert ("BTC/USDT", "KALMAN_MR_DUAL", "scale_phase") in plan
    
    # KALMAN_MR_DUAL should work with 'none' mode as well (like C method)
    plan = _build_run_plan(["BTC/USDT"], ["none"], ["KALMAN_MR_DUAL"])
    assert ("BTC/USDT", "KALMAN_MR_DUAL", "none") in plan
    
    # Mixed methods
    plan = _build_run_plan(["BTC/USDT"], ["scale_phase"], ["C", "S", "KALMAN_MR_DUAL"])
    assert ("BTC/USDT", "C", "scale_phase") in plan
    assert ("BTC/USDT", "S", "scale_phase") in plan
    assert ("BTC/USDT", "KALMAN_MR_DUAL", "scale_phase") in plan
