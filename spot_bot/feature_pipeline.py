"""
Feature pipeline for Spot Bot 2.0.

Transforms raw market data into model-ready features, including psi_logtime
representations used by mean reversion and Kalman stages.
"""

from typing import Any


class FeaturePipeline:
    """Builds reusable feature sets consumed by strategies and the regime engine."""

    def transform(self, market_data: Any) -> Any:
        """
        Convert raw market data into engineered features.
        This is where psi_logtime and other domain-specific transforms will live.
        """
        raise NotImplementedError("Feature transformation is not implemented yet.")
