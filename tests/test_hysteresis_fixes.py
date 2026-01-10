"""
Acceptance tests for hysteresis parameter plumbing and diagnostics fixes.

These tests verify:
- Test A: Parameter plumbing (hyst-k, hyst-floor affect delta_e_min)
- Test B: Hysteresis effect (hyst_floor changes trade count)
- Test C: Boundary condition (delta_e == delta_e_min suppresses)
- Test D: hyst-mode functionality
"""
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from spot_bot.core.hysteresis import apply_hysteresis, compute_hysteresis_threshold
from spot_bot.backtest import run_backtest


class TestParameterPlumbing:
    """Test A: Verify that changing hyst-k and hyst-floor affects delta_e_min values."""

    def test_hyst_k_affects_delta_e_min(self):
        """Higher hyst_k should produce higher delta_e_min."""
        rv_current = 0.05
        rv_ref = 0.05
        fee_rate = 0.001
        slippage_bps = 0.0
        spread_bps = 0.0
        hyst_floor = 0.02
        
        delta_e_min_low = compute_hysteresis_threshold(
            rv_current=rv_current,
            rv_ref=rv_ref,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            hyst_k=3.0,
            hyst_floor=hyst_floor,
        )
        
        delta_e_min_high = compute_hysteresis_threshold(
            rv_current=rv_current,
            rv_ref=rv_ref,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            hyst_k=10.0,
            hyst_floor=hyst_floor,
        )
        
        assert delta_e_min_high > delta_e_min_low, \
            f"Higher hyst_k should increase delta_e_min: {delta_e_min_high} vs {delta_e_min_low}"

    def test_hyst_floor_affects_delta_e_min(self):
        """Higher hyst_floor should produce higher delta_e_min (when floor is binding)."""
        rv_current = 0.01  # Low volatility to make floor binding
        rv_ref = 0.01
        fee_rate = 0.001
        slippage_bps = 0.0
        spread_bps = 0.0
        hyst_k = 5.0
        
        delta_e_min_low = compute_hysteresis_threshold(
            rv_current=rv_current,
            rv_ref=rv_ref,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            hyst_k=hyst_k,
            hyst_floor=0.02,
        )
        
        delta_e_min_high = compute_hysteresis_threshold(
            rv_current=rv_current,
            rv_ref=rv_ref,
            fee_rate=fee_rate,
            slippage_bps=slippage_bps,
            spread_bps=spread_bps,
            hyst_k=hyst_k,
            hyst_floor=0.06,
        )
        
        assert delta_e_min_high > delta_e_min_low, \
            f"Higher hyst_floor should increase delta_e_min: {delta_e_min_high} vs {delta_e_min_low}"
        # Note: Due to smooth max (tanh), the result might be slightly below the floor
        # but should be close to it
        assert delta_e_min_high >= 0.055, \
            f"delta_e_min should be close to hyst_floor: {delta_e_min_high} >= 0.055"


class TestBoundaryCondition:
    """Test C: Verify that delta_e == delta_e_min suppresses the trade."""

    def test_exact_boundary_suppresses(self):
        """When delta_e exactly equals delta_e_min, trade should be suppressed."""
        current_exposure = 0.10
        target_exposure = 0.15
        delta_e_min = 0.05  # Exactly the difference
        
        final_exposure, suppressed = apply_hysteresis(
            current_exposure=current_exposure,
            target_exposure=target_exposure,
            delta_e_min=delta_e_min,
            mode="exposure",
        )
        
        assert suppressed, "Trade should be suppressed when delta_e == delta_e_min"
        assert final_exposure == current_exposure, \
            f"Final exposure should equal current when suppressed: {final_exposure} == {current_exposure}"

    def test_below_boundary_suppresses(self):
        """When delta_e < delta_e_min, trade should be suppressed."""
        current_exposure = 0.10
        target_exposure = 0.14
        delta_e_min = 0.05  # Greater than the difference (0.04)
        
        final_exposure, suppressed = apply_hysteresis(
            current_exposure=current_exposure,
            target_exposure=target_exposure,
            delta_e_min=delta_e_min,
            mode="exposure",
        )
        
        assert suppressed, "Trade should be suppressed when delta_e < delta_e_min"
        assert final_exposure == current_exposure

    def test_above_boundary_allows(self):
        """When delta_e > delta_e_min, trade should be allowed."""
        current_exposure = 0.10
        target_exposure = 0.16
        delta_e_min = 0.05  # Less than the difference (0.06)
        
        final_exposure, suppressed = apply_hysteresis(
            current_exposure=current_exposure,
            target_exposure=target_exposure,
            delta_e_min=delta_e_min,
            mode="exposure",
        )
        
        assert not suppressed, "Trade should be allowed when delta_e > delta_e_min"
        assert final_exposure == target_exposure, \
            f"Final exposure should equal target when allowed: {final_exposure} == {target_exposure}"


class TestHysteresisMode:
    """Test D: Verify hyst-mode exposure works; zscore mode either works or raises clear error."""

    def test_exposure_mode_works(self):
        """Exposure mode should work as expected."""
        current_exposure = 0.10
        target_exposure = 0.20
        delta_e_min = 0.05
        
        final_exposure, suppressed = apply_hysteresis(
            current_exposure=current_exposure,
            target_exposure=target_exposure,
            delta_e_min=delta_e_min,
            mode="exposure",
        )
        
        assert not suppressed, "Large exposure change should not be suppressed"
        assert final_exposure == target_exposure

    def test_zscore_mode_works_when_zscore_available(self):
        """Z-score mode should work when z-scores are provided."""
        current_exposure = 0.10
        target_exposure = 0.20
        delta_e_min = 0.5  # z-score threshold
        current_zscore = -1.0
        target_zscore = -1.4  # delta_z = 0.4 < 0.5
        
        final_exposure, suppressed = apply_hysteresis(
            current_exposure=current_exposure,
            target_exposure=target_exposure,
            delta_e_min=delta_e_min,
            mode="zscore",
            current_zscore=current_zscore,
            target_zscore=target_zscore,
        )
        
        assert suppressed, "Small z-score change should be suppressed"
        assert final_exposure == current_exposure

    def test_zscore_mode_allows_large_zscore_change(self):
        """Z-score mode should allow trades when z-score change is large enough."""
        current_exposure = 0.10
        target_exposure = 0.20
        delta_e_min = 0.5  # z-score threshold
        current_zscore = -1.0
        target_zscore = -2.0  # delta_z = 1.0 > 0.5
        
        final_exposure, suppressed = apply_hysteresis(
            current_exposure=current_exposure,
            target_exposure=target_exposure,
            delta_e_min=delta_e_min,
            mode="zscore",
            current_zscore=current_zscore,
            target_zscore=target_zscore,
        )
        
        assert not suppressed, "Large z-score change should not be suppressed"
        assert final_exposure == target_exposure

    def test_invalid_mode_raises_error(self):
        """Invalid mode should raise a clear error."""
        with pytest.raises(ValueError, match="Invalid hysteresis mode.*invalid_mode"):
            apply_hysteresis(
                current_exposure=0.10,
                target_exposure=0.20,
                delta_e_min=0.05,
                mode="invalid_mode",
            )


class TestHysteresisEffect:
    """Test B: Verify that changing hyst_floor affects the number of executed trades in backtest."""

    def _create_test_data(self, rows: int = 100) -> pd.DataFrame:
        """Create test OHLCV data with small price oscillations."""
        import numpy as np
        
        timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
        base_price = 20000.0
        
        # Create oscillating prices with small changes (roughly 0.5% moves)
        # This should create exposure changes around 0.05 per step
        prices = base_price + 100 * np.sin(np.arange(rows) * 0.2) + np.random.randn(rows) * 10
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 10,
            "low": prices - 10,
            "close": prices,
            "volume": 1.0,
        })
        
        return df

    def test_higher_hyst_floor_reduces_trades(self):
        """Higher hyst_floor should reduce the number of executed trades."""
        df = self._create_test_data(rows=200)
        
        # Run with low hyst_floor
        equity_df_low, trades_df_low, summary_low = run_backtest(
            df=df,
            timeframe="1h",
            strategy_name="meanrev",
            psi_mode="scale_phase",
            psi_window=50,
            rv_window=50,
            conc_window=50,
            base=2.0,
            fee_rate=0.001,
            slippage_bps=0.0,
            max_exposure=0.3,
            initial_usdt=1000.0,
            hyst_k=5.0,
            hyst_floor=0.02,
            hyst_mode="exposure",
            log=False,
        )
        
        # Run with high hyst_floor
        equity_df_high, trades_df_high, summary_high = run_backtest(
            df=df,
            timeframe="1h",
            strategy_name="meanrev",
            psi_mode="scale_phase",
            psi_window=50,
            rv_window=50,
            conc_window=50,
            base=2.0,
            fee_rate=0.001,
            slippage_bps=0.0,
            max_exposure=0.3,
            initial_usdt=1000.0,
            hyst_k=5.0,
            hyst_floor=0.10,  # Much higher floor
            hyst_mode="exposure",
            log=False,
        )
        
        trades_low = len(trades_df_low)
        trades_high = len(trades_df_high)
        
        # Higher floor should result in fewer or equal trades (never more)
        assert trades_high <= trades_low, \
            f"Higher hyst_floor should reduce or maintain trades: {trades_high} <= {trades_low}"
        
        # Verify diagnostics are present in output
        assert "delta_e_min" in equity_df_low.columns, "delta_e_min should be in equity output"
        assert "suppressed" in equity_df_low.columns, "suppressed should be in equity output"

    def test_diagnostics_in_backtest_output(self):
        """Verify that diagnostic fields are present in backtest outputs."""
        df = self._create_test_data(rows=150)  # Increase rows for sufficient data
        
        equity_df, trades_df, summary = run_backtest(
            df=df,
            timeframe="1h",
            strategy_name="meanrev",
            psi_mode="scale_phase",
            psi_window=50,
            rv_window=50,
            conc_window=50,
            base=2.0,
            fee_rate=0.001,
            slippage_bps=0.0,
            max_exposure=0.3,
            initial_usdt=1000.0,
            hyst_k=5.0,
            hyst_floor=0.05,
            hyst_mode="exposure",
            log=False,
        )
        
        # Check equity DataFrame has diagnostic columns
        required_equity_cols = [
            "target_exposure_raw",
            "target_exposure_final",
            "delta_e",
            "delta_e_min",
            "suppressed",
            "clamped_long_only",
        ]
        for col in required_equity_cols:
            assert col in equity_df.columns, f"Equity DataFrame should contain {col}"
        
        # Check trades DataFrame has diagnostic columns (if there are trades)
        if not trades_df.empty:
            required_trade_cols = [
                "target_exposure_raw",
                "target_exposure_final",
                "delta_e",
                "delta_e_min",
                "suppressed",
                "clamped_long_only",
            ]
            for col in required_trade_cols:
                assert col in trades_df.columns, f"Trades DataFrame should contain {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
