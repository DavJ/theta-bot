import importlib


def test_feature_pipeline_imports_without_ccxt():
    try:
        module = importlib.import_module("spot_bot.features.feature_pipeline")
    except ModuleNotFoundError as e:
        assert "ccxt" not in str(e), "feature pipeline should not depend on ccxt at import time"
        raise
    assert hasattr(module, "compute_features")
