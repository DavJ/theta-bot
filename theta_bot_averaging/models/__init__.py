"""
Model zoo for theta bot: baseline classifiers and optional gating models.
"""

from .baseline import BaselineModel, PredictedOutput
from .gating import GRUGatingModel

__all__ = ["BaselineModel", "PredictedOutput", "GRUGatingModel"]
