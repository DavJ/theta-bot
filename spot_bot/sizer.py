"""
Thin re-export to the canonical sizer implementation.
"""

from spot_bot.portfolio.sizer import compute_target_position

# Backward-compatible alias for legacy imports.
PositionSizer = compute_target_position

__all__ = ["compute_target_position", "PositionSizer"]
