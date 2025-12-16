from __future__ import annotations

from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """
    Time-series split with optional purge and (currently ignored) embargo to avoid leakage.
    """

    def __init__(self, n_splits: int = 5, purge: int = 0, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo

    def split(self, X: Iterable) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        X = np.array(list(X))
        n_samples = len(X)
        fold_sizes = (n_samples // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1
        indices = np.arange(n_samples)
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_end = max(start - self.purge, 0)
            train_indices = indices[:train_end]

            current = stop
            yield train_indices, test_indices
