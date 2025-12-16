import numpy as np
import pandas as pd
import pytest

from theta_bot_averaging.models import BaselineModel


class _IdentityScaler:
    def fit(self, X):
        return self

    def transform(self, X):
        return X


class _StubModel:
    def __init__(self, probs):
        self.probs = probs
        self.classes_ = [-1, 0, 1]

    def predict_proba(self, X):
        return self.probs


def test_expected_return_and_thresholds():
    X = pd.DataFrame({"f": [0, 1]})
    probs = np.array([[0.1, 0.3, 0.6], [0.2, 0.4, 0.4]])

    model_high_thr = BaselineModel(positive_threshold=0.02, negative_threshold=-0.02)
    model_high_thr.scaler = _IdentityScaler()
    model_high_thr.model = _StubModel(probs)
    model_high_thr.class_mean_returns = {-1: -0.02, 0: 0.0, 1: 0.03}

    preds_high = model_high_thr.predict(X)
    assert preds_high.predicted_return.tolist() == pytest.approx([0.016, 0.008])
    assert preds_high.signal.eq(0).all()

    model_zero_thr = BaselineModel(positive_threshold=0.0, negative_threshold=0.0)
    model_zero_thr.scaler = _IdentityScaler()
    model_zero_thr.model = _StubModel(probs)
    model_zero_thr.class_mean_returns = {-1: -0.02, 0: 0.0, 1: 0.03}

    preds_zero = model_zero_thr.predict(X)
    assert preds_zero.signal.tolist() == [1, 1]
