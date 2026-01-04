"""Utility helpers for normalization and clipping."""

from typing import Union

Number = Union[float, int]


def clip01(value: Number) -> float:
    """Clip a numeric value to the inclusive [0, 1] range."""
    return max(0.0, min(1.0, float(value)))
