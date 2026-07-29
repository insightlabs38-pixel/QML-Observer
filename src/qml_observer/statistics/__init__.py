"""Statistics engine: framework-agnostic numerical primitives.

Milestone 3 (Volume IV). This package computes the statistical
primitives that the future detector layer (Milestone 4) builds on:
gradient-level reductions (`gradients.py`), loss-trend reductions
(`loss.py`), rolling-window bookkeeping (`rolling.py`, Issue #23),
SNR/measurement-uncertainty estimates (`snr.py`, Milestone 9), and
gradient-norm confidence intervals (`confidence.py`, Milestone 9,
Issue #69).

This module deliberately contains no detection or diagnosis logic of its
own -- see blueprint Volume VII ("detector outputs should not be directly
exposed as the final diagnosis") for why that separation matters.
"""

from qml_observer.statistics.confidence import (
    attach_gradient_norm_ci,
    bootstrap_gradient_norm_ci,
    estimate_gradient_norm_ci,
)
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
from qml_observer.statistics.rolling import RollingWindow
from qml_observer.statistics.snr import estimate_gradient_snr, estimate_measurement_uncertainty

__all__ = [
    "gradient_norm",
    "mean_absolute_gradient",
    "gradient_variance",
    "gradient_percentiles",
    "loss_slope",
    "relative_loss_improvement",
    "is_loss_stagnant",
    "RollingWindow",
    "estimate_gradient_snr",
    "estimate_measurement_uncertainty",
    "estimate_gradient_norm_ci",
    "bootstrap_gradient_norm_ci",
    "attach_gradient_norm_ci",
]
