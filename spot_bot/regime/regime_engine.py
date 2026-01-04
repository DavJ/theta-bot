"""Centralized regime decision logic (single source of truth)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from .types import RegimeDecision
from spot_bot.utils.normalization import clip01

MIN_SPAN = 1e-6
REDUCE_FACTOR = 0.8
RV_COLUMNS = ("rv", "RV")


class RegimeEngine:
    """
    Simple, configurable regime engine.

    Applies robust gating based on ensemble score (S) and volatility proxy (rv),
    producing a discrete risk_state and a smooth risk_budget in [0, 1].
    """

    def __init__(self, config: Dict[str, Any]):
        cfg = config or {}
        self.s_off = float(cfg.get("s_off", -0.1))
        self.s_on = float(cfg.get("s_on", 0.2))
        rv_off_value = cfg.get("rv_off")
        self.rv_off = float(rv_off_value) if rv_off_value is not None else None
        if self.rv_off is not None and self.rv_off <= 0.0:
            self.rv_off = None
        # Reduce threshold defaults below off threshold for a softer landing
        if self.rv_off is not None:
            default_rv_reduce: Optional[float] = self.rv_off * REDUCE_FACTOR
        else:
            default_rv_reduce = self.rv_off
        rv_reduce_value = cfg.get("rv_reduce", default_rv_reduce)
        self.rv_reduce = float(rv_reduce_value) if rv_reduce_value is not None else None
        self.s_budget_low = float(cfg.get("s_budget_low", self.s_off))
        self.s_budget_high = float(
            cfg.get("s_budget_high", self.s_on if self.s_on > self.s_off else self.s_off + MIN_SPAN)
        )
        guard_raw = cfg.get("rv_guard")
        self.rv_guard = float(guard_raw) if guard_raw is not None else self.rv_off

    def _validate_columns(self, features_df: pd.DataFrame) -> None:
        required = ["C", "S"]
        missing = [c for c in required if c not in features_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        if features_df.empty:
            raise ValueError("Feature DataFrame is empty.")

    def _extract_latest(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        self._validate_columns(features_df)
        latest = features_df.iloc[-1]
        if latest.isna().get("S", False) or latest.isna().get("C", False):
            raise ValueError("Latest feature row contains NaN for required columns.")

        rv_col = next((col for col in RV_COLUMNS if col in features_df.columns), None)
        rv_val = float(latest[rv_col]) if rv_col is not None and pd.notna(latest[rv_col]) else None

        return {
            "S": float(latest["S"]),
            "C": float(latest["C"]),
            "C_int": float(latest["C_int"]) if "C_int" in features_df.columns and pd.notna(latest["C_int"]) else None,
            "rv_col": rv_col,
            "rv": rv_val,
            "timestamp": latest.name,
        }

    def decide(self, features_df: pd.DataFrame) -> RegimeDecision:
        """
        Decide regime at the latest timestamp.

        Expects columns:
            - 'C'  (log-phase concentration)
            - 'S'  (ensemble score)
            - optional 'C_int'
            - optional 'rv' or 'RV' (volatility proxy)
        """
        latest = self._extract_latest(features_df)
        S = latest["S"]
        rv_val = latest["rv"]

        reasons = []
        risk_state = "ON"

        # OFF gating
        if S < self.s_off:
            risk_state = "OFF"
            reasons.append(f"S ({S:.4f}) < s_off ({self.s_off:.4f})")
        if self.rv_off is not None and rv_val is not None and rv_val > self.rv_off:
            risk_state = "OFF"
            reasons.append(f"rv ({rv_val:.4f}) > rv_off ({self.rv_off:.4f})")

        # REDUCE gating
        if risk_state == "ON":
            reduce_conditions = []
            if S < self.s_on:
                reduce_conditions.append(f"S ({S:.4f}) < s_on ({self.s_on:.4f})")
            if self.rv_reduce is not None and rv_val is not None and rv_val > self.rv_reduce:
                reduce_conditions.append(f"rv ({rv_val:.4f}) > rv_reduce ({self.rv_reduce:.4f})")
            if reduce_conditions:
                risk_state = "REDUCE"
                reasons.extend(reduce_conditions)

        # Risk budget mapping from score with optional volatility guard
        span = max(self.s_budget_high - self.s_budget_low, MIN_SPAN)
        budget_raw = (S - self.s_budget_low) / span
        budget = clip01(budget_raw)
        vol_guard = 1.0
        if rv_val is not None and self.rv_guard is not None and abs(self.rv_guard) > MIN_SPAN:
            vol_guard = clip01(1.0 - (rv_val / self.rv_guard))
            budget *= vol_guard

        reason = "; ".join(reasons) if reasons else "Within normal thresholds"

        diagnostics = {
            "timestamp": latest["timestamp"],
            "S": S,
            "C": latest["C"],
            "C_int": latest["C_int"],
            "rv": rv_val,
            "s_off": self.s_off,
            "s_on": self.s_on,
            "rv_off": self.rv_off,
            "rv_reduce": self.rv_reduce,
            "budget_raw": budget_raw,
            "vol_guard": vol_guard,
            "risk_state": risk_state,
        }

        return RegimeDecision(
            risk_state=risk_state, risk_budget=budget, reason=reason, diagnostics=diagnostics
        )
