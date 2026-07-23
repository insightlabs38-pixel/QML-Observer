"""Loss-trend statistical primitives.

Milestone 3, Volume IV (`statistics/loss.py`), Issues #21-#22:

- Issue #21: `loss_slope`
- Issue #22: `relative_loss_improvement`

`is_loss_stagnant` is included alongside these per the blueprint's
`statistics/loss.py` spec (Volume IV); it is a thin, explainable
convenience built directly on `relative_loss_improvement` and is exposed
here so the future `StagnationDetector` (Milestone 4, Issue #27) has a
single, independently-tested primitive to call rather than re-deriving
the stagnation check inline.

Per blueprint Volume VI ("Important: a small gradient alone should never
be enough to stop training") and addendum §7, this module never raises on
NaN/Inf *values* in the loss sequence -- those are meaningful signal from
a diverging optimizer, and classifying them (`IssueType.UNSTABLE`) is a
detector-layer concern. It does raise on structurally insufficient input
(fewer than two observations), since a trend cannot be computed from a
single point and treating that ambiguity as "no trend" would silently
misinform the detector layer.
"""

import math

import numpy as np


def _validate_losses(losses, caller: str) -> np.ndarray:
    """Coerce `losses` to a float array, requiring at least two points.

    A slope or relative-improvement figure is undefined for fewer than
    two observations; callers (the future rolling-window / detector
    layer) are expected to wait until enough history has accumulated
    before calling these functions.
    """
    array = np.asarray(losses, dtype=float)
    if array.ndim != 1:
        raise ValueError(
            f"{caller}() expects a 1-D sequence of loss values, got shape {array.shape}"
        )
    if array.size < 2:
        raise ValueError(
            f"{caller}() requires at least 2 loss values to compute a trend, got {array.size}"
        )
    return array


def loss_slope(losses) -> float:
    """Compute the least-squares slope of loss versus step index.

    Losses are assumed to be evenly spaced observations (e.g. one per
    training step, as they arrive in the rolling window) -- the step
    index used for the fit is simply `0, 1, ..., len(losses) - 1`, not
    any externally supplied step numbers. A negative slope indicates the
    loss is decreasing (healthy progress); a slope near zero indicates a
    flat loss curve (a signal for stagnation/plateau detectors, not proof
    of one on its own).

    Args:
        losses: Sequence of at least 2 loss values, oldest first.

    Returns:
        The fitted slope as a float. `nan` if the input contains NaN/Inf
        values (a diverging optimizer's slope is not a well-defined
        number -- that is itself the signal for the detector layer).

    Raises:
        ValueError: If fewer than 2 values are supplied.
    """
    array = _validate_losses(losses, "loss_slope")
    if not np.all(np.isfinite(array)):
        return float("nan")

    steps = np.arange(array.size, dtype=float)
    slope, _intercept = np.polyfit(steps, array, deg=1)
    return float(slope)


def relative_loss_improvement(losses) -> float:
    """Compute the relative loss improvement from the first to the last value.

    Defined as ``(losses[0] - losses[-1]) / abs(losses[0])``: positive
    means the loss got better (decreased), negative means it got worse.

    Args:
        losses: Sequence of at least 2 loss values, oldest first.

    Returns:
        The relative improvement as a float. If the baseline
        (`losses[0]`) is exactly `0.0`, returns `0.0` when there was no
        change at all, or a signed `inf` when there was a change from a
        zero baseline (both are honest descriptions of an edge case
        rather than a raised `ZeroDivisionError`). `nan` propagates if
        the input contains NaN.

    Raises:
        ValueError: If fewer than 2 values are supplied.
    """
    array = _validate_losses(losses, "relative_loss_improvement")
    baseline = array[0]
    final = array[-1]

    if math.isnan(baseline) or math.isnan(final):
        return float("nan")

    delta = baseline - final
    denom = abs(baseline)

    if denom == 0.0:
        if delta == 0.0:
            return 0.0
        return math.copysign(float("inf"), delta)

    return float(delta / denom)


def is_loss_stagnant(losses, threshold: float) -> bool:
    """Check whether recent loss improvement is below `threshold`.

    A thin convenience wrapper around `relative_loss_improvement`: the
    loss is considered stagnant if its *magnitude* of relative change is
    smaller than `threshold` (i.e. it has moved neither meaningfully
    better nor meaningfully worse).

    Args:
        losses: Sequence of at least 2 loss values, oldest first.
        threshold: Non-negative relative-improvement threshold below
            which the loss is considered stagnant.

    Returns:
        `True` if `abs(relative_loss_improvement(losses)) < threshold`,
        `False` otherwise. Returns `False` (not stagnant) if the
        improvement is undefined (`nan`) -- an undefined trend should
        never be silently reported as a stagnation signal.

    Raises:
        ValueError: If fewer than 2 loss values are supplied, or if
            `threshold` is negative.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    improvement = relative_loss_improvement(losses)
    if math.isnan(improvement):
        return False
    return abs(improvement) < threshold
