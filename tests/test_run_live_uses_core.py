"""
Test that run_live.py compute_step delegates to core engine.

This test verifies that compute_step no longer computes cost/hysteresis/rounding
locally but instead calls the unified core engine.
"""

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from spot_bot.features import FeatureConfig
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.run_live import compute_step
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


def test_compute_step_calls_core_adapter():
    """
    Test that compute_step delegates to compute_step_with_core_full.
    
    This verifies the refactor removed duplicated cost/hysteresis logic.
    """
    # Create minimal test data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "open": [50000.0] * 100,
        "high": [51000.0] * 100,
        "low": [49000.0] * 100,
        "close": [50000.0] * 100,
        "volume": [100.0] * 100,
    })
    
    feature_cfg = FeatureConfig(base=10.0)
    regime_engine = RegimeEngine({})
    strategy = MeanReversionStrategy()
    balances = {"btc": 0.0, "usdt": 1000.0}
    
    # Patch the core adapter to track calls
    with patch("spot_bot.core.legacy_adapter.compute_step_with_core_full") as mock_core:
        # Set up mock to return valid result
        from spot_bot.core.legacy_adapter import StepResultFromCore
        from spot_bot.core.types import TradePlan
        
        mock_result = StepResultFromCore(
            ts=pd.Timestamp("2024-01-01", tz="UTC"),
            close=50000.0,
            decision=Mock(risk_state="ON", risk_budget=1.0),
            intent=Mock(desired_exposure=0.5),
            target_exposure=0.5,
            target_btc=0.01,
            delta_btc=0.01,
            equity={"equity_usdt": 1000.0, "btc": 0.0, "usdt": 1000.0},
            execution=None,
            features_row=pd.Series({"S": 0.5, "C": 0.5, "rv": 0.02}),
            plan=TradePlan(
                action="BUY",
                target_exposure=0.5,
                target_base=0.01,
                delta_base=0.01,
                notional=500.0,
                exec_price_hint=50000.0,
                reason="planned",
                diagnostics={},
            ),
            diagnostics={},
        )
        mock_core.return_value = mock_result
        
        # Call compute_step
        try:
            result = compute_step(
                ohlcv_df=df,
                feature_cfg=feature_cfg,
                regime_engine=regime_engine,
                strategy=strategy,
                max_exposure=0.5,
                fee_rate=0.001,
                balances=balances,
                mode="dryrun",
                broker=None,
                slippage_bps=5.0,
                spread_bps=2.0,
                hyst_k=5.0,
                hyst_floor=0.02,
                hyst_mode="exposure",
            )
        except Exception:
            # If features computation fails (not enough data), that's OK for this test
            # We just want to verify the core adapter was called
            pass
        
        # Verify core adapter was called
        assert mock_core.called, "compute_step should call compute_step_with_core_full"
        
        # Verify it was called with correct params
        call_args = mock_core.call_args
        assert call_args is not None
        assert call_args[1]["fee_rate"] == 0.001
        assert call_args[1]["slippage_bps"] == 5.0
        assert call_args[1]["hyst_k"] == 5.0


def test_compute_step_no_local_cost_computation():
    """
    Verify compute_step doesn't compute cost locally.
    
    The old implementation had:
        cost = fee_rate + 2 * (slippage_bps / 10_000.0) + (spread_bps / 10_000.0)
    
    This should no longer exist in the new implementation.
    """
    import inspect
    source = inspect.getsource(compute_step)
    
    # Check that the old cost computation is not present
    assert "cost = fee_rate + 2 *" not in source, \
        "compute_step should not compute cost locally"
    assert "delta_e_min = max(hyst_floor, hyst_k * cost" not in source, \
        "compute_step should not compute hysteresis threshold locally"


def test_compute_step_no_local_rv_ref_computation():
    """
    Verify compute_step doesn't compute rv_ref locally.
    
    The old implementation had:
        rv_ref_candidates = rv_series.tail(500)
        rv_ref = float(rv_ref_candidates.median())
    
    This should be delegated to core now.
    """
    import inspect
    source = inspect.getsource(compute_step)
    
    # Check that the old rv_ref computation is not present
    assert "rv_ref_candidates = rv_series.tail(500)" not in source, \
        "compute_step should not compute rv_ref locally"
    assert "rv_ref_candidates.median()" not in source, \
        "compute_step should delegate rv_ref to core"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
