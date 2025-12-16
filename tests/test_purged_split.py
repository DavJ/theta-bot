import numpy as np

from theta_bot_averaging.validation import PurgedTimeSeriesSplit


def test_purged_split_removes_overlap():
    splitter = PurgedTimeSeriesSplit(n_splits=3, purge=1, embargo=1)
    X = np.arange(9)
    for train_idx, test_idx in splitter.split(X):
        # No overlap
        assert set(train_idx).isdisjoint(set(test_idx))
        if len(train_idx) > 0:
            assert train_idx.max() < test_idx.min()
