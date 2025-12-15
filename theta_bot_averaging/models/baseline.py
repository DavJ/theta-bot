from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


PredictionMode = Literal["logit", "mlp"]


@dataclass
class PredictedOutput:
    predicted_return: pd.Series
    signal: pd.Series
    probabilities: Optional[pd.DataFrame] = None


class BaselineModel:
    """
    Baseline classification model (logistic regression or small MLP).
    Produces predicted_return and discrete signal in {-1, 0, 1}.
    """

    def __init__(
        self,
        mode: PredictionMode = "logit",
        positive_threshold: float = 0.0005,
        negative_threshold: float = -0.0005,
        random_state: int = 42,
        max_iter: int = 200,
    ):
        self.mode = mode
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.random_state = random_state
        self.max_iter = max_iter

        self.scaler = StandardScaler()
        if mode == "logit":
            self.model = LogisticRegression(
                multi_class="multinomial",
                max_iter=max_iter,
                n_jobs=None,
                random_state=random_state,
            )
        elif mode == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                alpha=1e-4,
                batch_size=64,
                max_iter=max_iter,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.scaler.fit(X.values)
        Xs = self.scaler.transform(X.values)
        if len(np.unique(y.values)) < 2:
            # fallback to constant predictor to keep pipeline running
            self.model = DummyClassifier(strategy="most_frequent")
            self.model.fit(Xs, y.values)
        else:
            self.model.fit(Xs, y.values)

    def predict(self, X: pd.DataFrame) -> PredictedOutput:
        Xs = self.scaler.transform(X.values)
        proba = self.model.predict_proba(Xs)
        classes = list(self.model.classes_)
        # Map class probabilities to expectation (+1 * p_pos + -1 * p_neg)
        proba_df = pd.DataFrame(proba, index=X.index, columns=classes)

        # Ensure columns for -1, 0, 1 exist
        p_pos = proba_df.get(1, pd.Series(0, index=X.index))
        p_neg = proba_df.get(-1, pd.Series(0, index=X.index))
        expected_return = p_pos - p_neg

        predicted_return = expected_return
        signal = predicted_return.copy()
        signal[:] = 0
        signal[predicted_return > self.positive_threshold] = 1
        signal[predicted_return < self.negative_threshold] = -1

        return PredictedOutput(
            predicted_return=predicted_return,
            signal=signal,
            probabilities=proba_df,
        )
