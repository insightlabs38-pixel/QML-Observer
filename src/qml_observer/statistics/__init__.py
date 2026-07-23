"""Statistics engine: framework-agnostic numerical primitives.

Milestone 3 (Volume IV). This package computes the statistical
primitives that the future detector layer (Milestone 4) builds on:
gradient-level reductions (`gradients.py`), loss-trend reductions
(`loss.py`), rolling-window bookkeeping (`rolling.py`, Issue #23), and
SNR/measurement-uncertainty estimates (`snr.py`).

This module deliberately contains no detection or diagnosis logic of its
own -- see blueprint Volume VII ("detector outputs should not be directly
exposed as the final diagnosis") for why that separation matters.
"""

from qml_observer.statistics.gradients import (
    gradient_norm,
    gradient_percentiles,
    gradient_variance,
    mean_absolute_gradient,
)
from qml_observer.statistics.loss import (
    is_loss_stagnant,
    loss_slope,
    relative_loss_improvement,
)

__all__ = [
    "gradient_norm",
    "mean_absolute_gradient",
    "gradient_variance",
    "gradient_percentiles",
    "loss_slope",
    "relative_loss_improvement",
    "is_loss_stagnant",
]
