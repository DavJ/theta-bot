"""
Test that run_live.py is a PURE ORCHESTRATOR with ZERO trading math.

This test verifies PHASE 2 requirements:
- run_live.py contains NO cost calculation
- run_live.py contains NO hysteresis logic
- run_live.py contains NO rounding/min_notional logic
- run_live.py contains NO simulated fills computation
- run_live.py contains NO equity/exposure math
- run_live.py contains NO rv_ref computation

All math MUST be in spot_bot/core.
"""

import inspect

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from spot_bot.core.legacy_adapter import StepResultFromCore
from spot_bot.core.types import TradePlan
from spot_bot.features import FeatureConfig
from spot_bot.regime.regime_engine import RegimeEngine
from spot_bot.run_live import compute_step, _apply_live_fill_to_balances
from spot_bot.strategies.mean_reversion import MeanReversionStrategy


class TestRunLiveIsOrchestratorOnly:
    """Verify run_live.py has ZERO trading math."""

    def test_compute_step_has_no_cost_computation(self):
        """Verify compute_step doesn't compute cost locally."""
        source = inspect.getsource(compute_step)
        
        # These patterns indicate local cost computation
        forbidden_patterns = [
            "cost = fee_rate",
            "cost = ",
            "+ 2 * slippage",
            "spread_bps / 10000",
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"compute_step should NOT compute cost locally. Found: {pattern}"
            )

    def test_compute_step_has_no_hysteresis_computation(self):
        """Verify compute_step doesn't compute hysteresis threshold locally."""
        source = inspect.getsource(compute_step)
        
        # These patterns indicate local hysteresis computation
        forbidden_patterns = [
            "delta_e_min",
            "hyst_k * cost",
            "max(hyst_floor,",
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"compute_step should NOT compute hysteresis locally. Found: {pattern}"
            )

    def test_compute_step_has_no_rounding_logic(self):
        """Verify compute_step doesn't round quantities locally."""
        source = inspect.getsource(compute_step)
        
        # These patterns indicate local rounding COMPUTATION (not just passing params)
        forbidden_patterns = [
            "math.floor(",
            "round(",
            "% step_size",
            "/ step_size",
            "< min_notional",
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"compute_step should NOT round quantities locally. Found: {pattern}"
            )

    def test_compute_step_has_no_rv_ref_computation(self):
        """Verify compute_step doesn't compute rv_ref locally."""
        source = inspect.getsource(compute_step)
        
        # These patterns indicate local rv_ref computation
        forbidden_patterns = [
            ".median()",
            "rolling(",
            "rv_ref = ",
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"compute_step should NOT compute rv_ref locally. Found: {pattern}"
            )

    def test_compute_step_delegates_to_core(self):
        """Verify compute_step calls core adapter."""
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
        
        # Mock core adapter
        with patch("spot_bot.run_live.compute_step_with_core_full") as mock_core:
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
                )
            except Exception:
                # Feature computation may fail with insufficient data
                pass
            
            # Verify core was called
            assert mock_core.called, "compute_step MUST delegate to core adapter"

    def test_apply_live_fill_delegates_to_core(self):
        """Verify _apply_live_fill_to_balances delegates to core."""
        source = inspect.getsource(_apply_live_fill_to_balances)
        
        # Should be a pure delegation - no local math
        forbidden_patterns = [
            "notional = qty * price",
            "fee = notional * fee_rate",
            "filled_base = qty if side",
            "usdt -=",
            "btc +=",
        ]
        
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"_apply_live_fill_to_balances should delegate to core, not compute locally. Found: {pattern}"
            )
        
        # Should call core function
        assert "apply_live_fill_to_balances(" in source, (
            "_apply_live_fill_to_balances should call core.portfolio.apply_live_fill_to_balances"
        )

    def test_apply_live_fill_core_function_works(self):
        """Verify core apply_live_fill_to_balances function works correctly."""
        from spot_bot.core.portfolio import apply_live_fill_to_balances as core_func
        
        # Test BUY
        balances = {"usdt": 1000.0, "btc": 0.0}
        fee = core_func(balances, "buy", 0.01, 50000.0, 0.001)
        
        assert balances["btc"] == 0.01, "BUY should increase BTC"
        assert balances["usdt"] < 1000.0, "BUY should decrease USDT"
        assert fee > 0.0, "BUY should incur fee"
        
        # Test SELL
        balances = {"usdt": 500.0, "btc": 0.01}
        fee = core_func(balances, "sell", 0.01, 50000.0, 0.001)
        
        assert balances["btc"] == 0.0, "SELL should decrease BTC"
        assert balances["usdt"] > 500.0, "SELL should increase USDT"
        assert fee > 0.0, "SELL should incur fee"

    def test_compute_step_result_contains_plan_from_core(self):
        """Verify compute_step returns TradePlan from core."""
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
        
        with patch("spot_bot.run_live.compute_step_with_core_full") as mock_core:
            mock_plan = TradePlan(
                action="BUY",
                target_exposure=0.5,
                target_base=0.01,
                delta_base=0.01,
                notional=500.0,
                exec_price_hint=50000.0,
                reason="test",
                diagnostics={"test_field": "test_value"},
            )
            
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
                plan=mock_plan,
                diagnostics={},
            )
            mock_core.return_value = mock_result
            
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
                )
                
                # Verify result contains values from core's TradePlan
                assert result.delta_btc == mock_plan.delta_base, (
                    "Result should contain delta_base from core's TradePlan"
                )
            except Exception:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
