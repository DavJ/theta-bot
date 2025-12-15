from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class GatingOutput:
    gate: pd.Series
    signal: pd.Series


class GRUGatingModel:
    """
    Lightweight gating model placeholder.

    For minimal dependency footprint, this implementation uses a simple
    exponential moving average on recent baseline signals to emulate
    gating behavior. If PyTorch is available, the class will optionally
    use it; otherwise it falls back to deterministic gating.
    """

    def __init__(self, window: int = 32, threshold: float = 0.2):
        self.window = window
        self.threshold = threshold
        self._use_torch = False
        try:
            import torch  # type: ignore

            self._use_torch = True
            self._torch = torch
        except ImportError:
            self._use_torch = False
            self._torch = None

    def fit(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        # Deterministic placeholder: nothing to fit without torch.
        if self._use_torch:
            # Minimal GRU with binary output; kept tiny to avoid overfitting.
            torch = self._torch
            input_dim = sequences.shape[-1]
            hidden_size = 32
            self._gru = torch.nn.GRU(
                input_dim, hidden_size, num_layers=1, batch_first=True, dropout=0.1
            )
            self._head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1), torch.nn.Sigmoid()
            )
            optim = torch.optim.Adam(
                list(self._gru.parameters()) + list(self._head.parameters()), lr=1e-3
            )
            loss_fn = torch.nn.BCELoss()
            X = torch.tensor(sequences, dtype=torch.float32)
            y = torch.tensor(labels.reshape(-1, 1), dtype=torch.float32)
            self._gru.train()
            self._head.train()
            for _ in range(10):
                optim.zero_grad()
                out, _ = self._gru(X)
                gate = self._head(out[:, -1, :])
                loss = loss_fn(gate, y)
                loss.backward()
                optim.step()

    def predict(self, sequences: np.ndarray, baseline_signal: pd.Series) -> GatingOutput:
        idx = baseline_signal.index
        if self._use_torch and hasattr(self, "_gru"):
            torch = self._torch
            X = torch.tensor(sequences, dtype=torch.float32)
            self._gru.eval()
            self._head.eval()
            with torch.no_grad():
                out, _ = self._gru(X)
                gate = self._head(out[:, -1, :]).squeeze().cpu().numpy()
        else:
            # Deterministic gating based on recent baseline signal volatility.
            rolling = baseline_signal.rolling(self.window).mean().fillna(0.0)
            gate = (rolling.abs() / (rolling.abs().rolling(self.window).max() + 1e-9)).clip(
                0, 1
            )

        gate_series = pd.Series(gate, index=idx).clip(0, 1)
        gated_signal = baseline_signal * (gate_series > self.threshold)
        return GatingOutput(gate=gate_series, signal=gated_signal)
