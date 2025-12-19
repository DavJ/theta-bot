from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

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

        self.class_mean_returns: Dict[int, float] = self._default_class_mean_returns()
        self.scaler = StandardScaler()
        if mode == "logit":
            self.model = LogisticRegression(
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

    def fit(self, X: pd.DataFrame, y: pd.Series, future_return: Optional[pd.Series] = None) -> None:
        self.scaler.fit(X.values)
        Xs = self.scaler.transform(X.values)
        if len(np.unique(y.values)) < 2:
            # fallback to constant predictor to keep pipeline running
            self.model = DummyClassifier(strategy="most_frequent")
            self.model.fit(Xs, y.values)
        else:
            self.model.fit(Xs, y.values)

        if future_return is not None:
            aligned = future_return.reindex(y.index)
            df = pd.DataFrame({"label": y, "future_return": aligned}).dropna(subset=["future_return"])
            means = df.groupby("label")["future_return"].mean()
            self.class_mean_returns = self._default_class_mean_returns()
            for cls, val in means.items():
                self.class_mean_returns[int(cls)] = float(val)
            
            # Sanity check: class mean returns should satisfy mu[-1] < mu[0] < mu[+1]
            # This ensures that labels are correctly aligned with future returns
            mu_neg = self.class_mean_returns.get(-1)
            mu_zero = self.class_mean_returns.get(0)
            mu_pos = self.class_mean_returns.get(1)
            
            violations = []
            if mu_neg is not None and mu_zero is not None and mu_neg >= mu_zero:
                violations.append(f"mu[-1]={mu_neg:.6f} >= mu[0]={mu_zero:.6f}")
            if mu_zero is not None and mu_pos is not None and mu_zero >= mu_pos:
                violations.append(f"mu[0]={mu_zero:.6f} >= mu[+1]={mu_pos:.6f}")
            if mu_neg is not None and mu_pos is not None and mu_neg >= mu_pos:
                violations.append(f"mu[-1]={mu_neg:.6f} >= mu[+1]={mu_pos:.6f}")
            
            if violations:
                print("\n" + "=" * 70)
                print("⚠️  WARNING: Class mean return ordering violation detected!")
                print("=" * 70)
                print("Expected: mu[-1] < mu[0] < mu[+1]")
                print(f"Actual: mu[-1]={mu_neg}, mu[0]={mu_zero}, mu[+1]={mu_pos}")
                print("\nViolations:")
                for v in violations:
                    print(f"  - {v}")
                print("\nThis indicates a potential issue with:")
                print("  1. Label construction (sign inversion)")
                print("  2. Future return computation (wrong direction)")
                print("  3. Data quality (unrealistic or corrupted)")
                print("\nSample of (future_return, label) pairs for inspection:")
                sample = df[["future_return", "label"]].head(20)
                print(sample.to_string())
                print("=" * 70 + "\n")

    def predict(self, X: pd.DataFrame) -> PredictedOutput:
        Xs = self.scaler.transform(X.values)
        proba = self.model.predict_proba(Xs)
        classes = list(self.model.classes_)
        # Map class probabilities to expectation (+1 * p_pos + -1 * p_neg)
        proba_df = pd.DataFrame(proba, index=X.index, columns=classes)

        # Ensure columns for -1, 0, 1 exist
        p_pos = proba_df.get(1, pd.Series(0, index=X.index))
        p_neg = proba_df.get(-1, pd.Series(0, index=X.index))
        p_neutral = proba_df.get(0, pd.Series(0, index=X.index))
        expected_return = (
            p_pos * self.class_mean_returns.get(1, 0.0)
            + p_neg * self.class_mean_returns.get(-1, 0.0)
            + p_neutral * self.class_mean_returns.get(0, 0.0)
        )

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

    @staticmethod
    def _default_class_mean_returns() -> Dict[int, float]:
        return {-1: 0.0, 0: 0.0, 1: 0.0}
