"""
Dual-Stream Model: Theta + Mellin(Theta) fusion with optional PyTorch.

This module implements a model that processes two input streams:
1. Theta stream: sequence of theta-reconstructed signals (via GRU or 1D CNN)
2. Mellin stream: compact Mellin transform features (via MLP)

The streams are fused with gating and produce trading signals {-1, 0, 1}.
Falls back to BaselineModel when PyTorch is unavailable.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .baseline import BaselineModel, PredictedOutput


class DualStreamModel:
    """
    Dual-stream model with Theta sequence processing and Mellin feature fusion.

    Uses PyTorch if available for GRU-based sequence modeling and gated fusion.
    Falls back to BaselineModel on flattened features when PyTorch is unavailable.
    """

    def __init__(
        self,
        positive_threshold: float = 0.0005,
        negative_threshold: float = -0.0005,
        random_state: int = 42,
        device: str = "cpu",
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        hidden_dim: int = 64,
        mellin_dim: int = 32,
        dropout: float = 0.2,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize DualStreamModel.

        Parameters
        ----------
        positive_threshold : float
            Threshold for positive trading signal (buy)
        negative_threshold : float
            Threshold for negative trading signal (sell)
        random_state : int
            Random seed for reproducibility
        device : str
            PyTorch device ("cpu" or "cuda")
        epochs : int
            Number of training epochs (PyTorch mode)
        batch_size : int
            Training batch size (PyTorch mode)
        lr : float
            Learning rate (PyTorch mode)
        hidden_dim : int
            Hidden dimension for theta sequence encoder (PyTorch mode)
        mellin_dim : int
            Hidden dimension for Mellin MLP (PyTorch mode)
        dropout : float
            Dropout probability for regularization
        weight_decay : float
            L2 regularization weight decay
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.random_state = random_state
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.mellin_dim = mellin_dim
        self.dropout = dropout
        self.weight_decay = weight_decay

        self.class_mean_returns: Dict[int, float] = self._default_class_mean_returns()

        # Try to import PyTorch
        self._use_torch = False
        self._torch = None
        self._model = None
        self._fallback_model = None

        try:
            import torch  # type: ignore

            self._use_torch = True
            self._torch = torch
            # Set random seed
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)
            np.random.seed(random_state)
        except ImportError:
            self._use_torch = False

    def fit(
        self,
        X_theta: np.ndarray,
        X_mellin: np.ndarray,
        y: np.ndarray,
        future_return: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit the dual-stream model.

        Parameters
        ----------
        X_theta : np.ndarray
            Theta sequence input, shape (N, window) or (N, window, C)
        X_mellin : np.ndarray
            Mellin features, shape (N, F)
        y : np.ndarray
            Target labels {-1, 0, 1}, shape (N,)
        future_return : np.ndarray, optional
            Future returns for computing class mean returns, shape (N,)
        """
        if self._use_torch:
            self._fit_torch(X_theta, X_mellin, y, future_return)
        else:
            self._fit_fallback(X_theta, X_mellin, y, future_return)

    def predict(
        self, X_theta: np.ndarray, X_mellin: np.ndarray, index: pd.DatetimeIndex
    ) -> PredictedOutput:
        """
        Predict trading signals and expected returns.

        Parameters
        ----------
        X_theta : np.ndarray
            Theta sequence input, shape (N, window) or (N, window, C)
        X_mellin : np.ndarray
            Mellin features, shape (N, F)
        index : pd.DatetimeIndex
            Timestamps for predictions

        Returns
        -------
        PredictedOutput
            Contains predicted_return, signal, and probabilities
        """
        if self._use_torch and self._model is not None:
            return self._predict_torch(X_theta, X_mellin, index)
        else:
            return self._predict_fallback(X_theta, X_mellin, index)

    def _fit_torch(
        self,
        X_theta: np.ndarray,
        X_mellin: np.ndarray,
        y: np.ndarray,
        future_return: Optional[np.ndarray],
    ) -> None:
        """Fit using PyTorch (GRU + MLP with gating)."""
        torch = self._torch

        # Ensure X_theta is 3D (N, window, C)
        if X_theta.ndim == 2:
            X_theta = X_theta[:, :, np.newaxis]  # Add channel dim

        N, window, C = X_theta.shape
        F = X_mellin.shape[1]

        # Map labels to 0, 1, 2 for CrossEntropyLoss
        y_mapped = y.copy()
        y_mapped[y == -1] = 0
        y_mapped[y == 0] = 1
        y_mapped[y == 1] = 2

        # Build model
        self._model = self._build_torch_model(window, C, F)
        self._model.to(self.device)

        # Prepare data
        X_theta_t = torch.tensor(X_theta, dtype=torch.float32, device=self.device)
        X_mellin_t = torch.tensor(X_mellin, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_mapped, dtype=torch.long, device=self.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        self._model.train()
        dataset = torch.utils.data.TensorDataset(X_theta_t, X_mellin_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_theta, batch_mellin, batch_y in loader:
                optimizer.zero_grad()
                logits = self._model(batch_theta, batch_mellin)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Compute class mean returns
        if future_return is not None:
            self._compute_class_mean_returns(y, future_return)

    def _predict_torch(
        self, X_theta: np.ndarray, X_mellin: np.ndarray, index: pd.DatetimeIndex
    ) -> PredictedOutput:
        """Predict using PyTorch model."""
        torch = self._torch

        # Ensure X_theta is 3D
        if X_theta.ndim == 2:
            X_theta = X_theta[:, :, np.newaxis]

        X_theta_t = torch.tensor(X_theta, dtype=torch.float32, device=self.device)
        X_mellin_t = torch.tensor(X_mellin, dtype=torch.float32, device=self.device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_theta_t, X_mellin_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        # Map back: class 0 -> label -1, class 1 -> label 0, class 2 -> label 1
        classes = [-1, 0, 1]
        proba_df = pd.DataFrame(proba, index=index, columns=classes)

        # Compute expected return
        p_neg = proba_df[-1]
        p_neutral = proba_df[0]
        p_pos = proba_df[1]

        expected_return = (
            p_neg * self.class_mean_returns.get(-1, 0.0)
            + p_neutral * self.class_mean_returns.get(0, 0.0)
            + p_pos * self.class_mean_returns.get(1, 0.0)
        )

        # Generate signal from thresholds
        signal = pd.Series(0, index=index)
        signal[expected_return > self.positive_threshold] = 1
        signal[expected_return < self.negative_threshold] = -1

        return PredictedOutput(
            predicted_return=expected_return, signal=signal, probabilities=proba_df
        )

    def _fit_fallback(
        self,
        X_theta: np.ndarray,
        X_mellin: np.ndarray,
        y: np.ndarray,
        future_return: Optional[np.ndarray],
    ) -> None:
        """Fallback to BaselineModel when PyTorch unavailable."""
        # Flatten X_theta and concatenate with X_mellin
        if X_theta.ndim == 3:
            X_theta = X_theta.reshape(X_theta.shape[0], -1)
        X_combined = np.concatenate([X_theta, X_mellin], axis=1)

        # Create feature names
        n_theta_feats = X_theta.shape[1]
        n_mellin_feats = X_mellin.shape[1]
        theta_names = [f"theta_{i}" for i in range(n_theta_feats)]
        mellin_names = [f"mellin_{i}" for i in range(n_mellin_feats)]
        feature_names = theta_names + mellin_names

        # Create dataframe with dummy index (BaselineModel expects DataFrame)
        X_df = pd.DataFrame(X_combined, columns=feature_names)
        y_series = pd.Series(y)

        future_return_series = None
        if future_return is not None:
            future_return_series = pd.Series(future_return)

        # Use BaselineModel
        self._fallback_model = BaselineModel(
            mode="logit",
            positive_threshold=self.positive_threshold,
            negative_threshold=self.negative_threshold,
            random_state=self.random_state,
        )
        self._fallback_model.fit(X_df, y_series, future_return=future_return_series)
        self.class_mean_returns = self._fallback_model.class_mean_returns

    def _predict_fallback(
        self, X_theta: np.ndarray, X_mellin: np.ndarray, index: pd.DatetimeIndex
    ) -> PredictedOutput:
        """Predict using fallback BaselineModel."""
        # Flatten and combine
        if X_theta.ndim == 3:
            X_theta = X_theta.reshape(X_theta.shape[0], -1)
        X_combined = np.concatenate([X_theta, X_mellin], axis=1)

        n_theta_feats = X_theta.shape[1]
        n_mellin_feats = X_mellin.shape[1]
        theta_names = [f"theta_{i}" for i in range(n_theta_feats)]
        mellin_names = [f"mellin_{i}" for i in range(n_mellin_feats)]
        feature_names = theta_names + mellin_names

        X_df = pd.DataFrame(X_combined, columns=feature_names, index=index)

        return self._fallback_model.predict(X_df)

    def _build_torch_model(self, window: int, C: int, F: int):
        """Build PyTorch model architecture."""
        torch = self._torch
        nn = torch.nn

        class DualStreamNet(nn.Module):
            def __init__(
                self,
                window,
                C,
                F,
                hidden_dim,
                mellin_dim,
                dropout,
            ):
                super().__init__()

                # Theta branch: GRU for sequence processing
                self.theta_gru = nn.GRU(
                    input_size=C,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,  # Only 1 layer, so no dropout here
                )
                self.theta_dropout = nn.Dropout(dropout)

                # Mellin branch: MLP
                self.mellin_mlp = nn.Sequential(
                    nn.Linear(F, mellin_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mellin_dim, mellin_dim),
                    nn.ReLU(),
                )

                # Gating mechanism
                self.gate_net = nn.Sequential(
                    nn.Linear(mellin_dim, 1), nn.Sigmoid()
                )

                # Fusion and classification head
                fused_dim = hidden_dim + mellin_dim
                self.head = nn.Sequential(
                    nn.Linear(fused_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 3),  # 3 classes: -1, 0, 1
                )

            def forward(self, x_theta, x_mellin):
                # Theta branch
                _, h_n = self.theta_gru(x_theta)  # h_n: (1, batch, hidden_dim)
                theta_emb = h_n.squeeze(0)  # (batch, hidden_dim)
                theta_emb = self.theta_dropout(theta_emb)

                # Mellin branch
                mellin_emb = self.mellin_mlp(x_mellin)  # (batch, mellin_dim)

                # Gating
                gate = self.gate_net(mellin_emb)  # (batch, 1)
                gated_theta = gate * theta_emb  # (batch, hidden_dim)

                # Fusion
                fused = torch.cat([gated_theta, mellin_emb], dim=1)

                # Classification
                logits = self.head(fused)
                return logits

        return DualStreamNet(
            window, C, F, self.hidden_dim, self.mellin_dim, self.dropout
        )

    def _compute_class_mean_returns(
        self, y: np.ndarray, future_return: np.ndarray
    ) -> None:
        """Compute mean return per class for expected return calculation."""
        df = pd.DataFrame({"label": y, "future_return": future_return})
        df = df.dropna(subset=["future_return"])
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

    @staticmethod
    def _default_class_mean_returns() -> Dict[int, float]:
        return {-1: 0.0, 0: 0.0, 1: 0.0}
