"""
Derivatives State Drift Module.

Compute deterministic directional pressure (drift) from derivatives market data.
"""

from theta_bot_averaging.derivatives_state.loaders import (
    load_spot_series,
    load_funding_series,
    load_oi_series,
    load_basis_series,
)
from theta_bot_averaging.derivatives_state.features import (
    compute_zscore,
    compute_oi_change,
)
from theta_bot_averaging.derivatives_state.drift import (
    compute_drift,
    compute_determinism,
)
from theta_bot_averaging.derivatives_state.gating import (
    apply_quantile_gate,
    apply_threshold_gate,
    apply_combined_gate,
)
from theta_bot_averaging.derivatives_state.report import (
    generate_top_timestamps_report,
)

__all__ = [
    "load_spot_series",
    "load_funding_series",
    "load_oi_series",
    "load_basis_series",
    "compute_zscore",
    "compute_oi_change",
    "compute_drift",
    "compute_determinism",
    "apply_quantile_gate",
    "apply_threshold_gate",
    "apply_combined_gate",
    "generate_top_timestamps_report",
]
