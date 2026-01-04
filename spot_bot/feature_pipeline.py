from spot_bot.features.feature_pipeline import FeatureConfig, compute_features


class FeaturePipeline:
    """Thin wrapper around spot_bot.features.feature_pipeline for backward compatibility."""

    def transform(self, market_data, config: FeatureConfig | None = None):
        cfg = config or FeatureConfig()
        return compute_features(market_data, cfg)


__all__ = ["FeatureConfig", "compute_features", "FeaturePipeline"]
