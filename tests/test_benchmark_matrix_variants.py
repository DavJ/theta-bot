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
