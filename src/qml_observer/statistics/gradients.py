"""Gradient-level statistical primitives.

Milestone 3, Volume IV (`statistics/gradients.py`), Issues #18-#20:

- Issue #18: `gradient_norm`
- Issue #19: `gradient_variance`
- Issue #20: `gradient_percentiles`

`mean_absolute_gradient` is included alongside these per the blueprint's
`statistics/gradients.py` spec (Volume IV); it backs `GradientSnapshot.mean_abs`
and is exposed here as a standalone primitive for detectors that want it
without constructing a full snapshot.

These functions operate on raw arrays (or anything `np.asarray` accepts),
independent of `GradientSnapshot`. `summarize_gradient` (Milestone 1,
`schemas/gradient.py`) already computes the same reductions inline for the
single-step-snapshot case; this module exists so the detector layer
(Milestone 4) and rolling-window statistics (Issue #23) have reusable,
independently-tested primitives rather than reaching into a snapshot's
internals or re-deriving these reductions themselves.

Numerical edge cases (addendum §7): an empty gradient array almost always
indicates an adapter misconfiguration, so every function here raises a
clear `ValueError` rather than letting numpy raise an opaque one (e.g.
"zero-size array to reduction operation"). NaN/Inf values, by contrast,
are tolerated and propagated -- they are meaningful signal from a
diverging or degenerate gradient, not a data-quality error, and
classifying them is a detector-layer concern (Milestone 4), not this
module's.
"""

import numpy as np


def _flatten_and_validate(gradients, caller: str) -> np.ndarray:
    """Coerce `gradients` to a flat float array, rejecting empty input.

    Shared by every function in this module so that the "empty array"
    error message and behavior stay consistent with
    `schemas.gradient.summarize_gradient` (addendum §7).
    """
    array = np.asarray(gradients, dtype=float)
    if array.size == 0:
        raise ValueError(
            f"{caller}() received an empty gradient array. This usually "
            "indicates a misconfigured adapter or a training loop that "
            "failed to compute gradients for this step."
        )
    return array.ravel()


def gradient_norm(gradients, ord: int = 2) -> float:
    """Compute the p-norm of a (flattened) gradient array.

    Args:
        gradients: Array-like of gradient values, any shape.
        ord: Order of the norm passed through to `numpy.linalg.norm`
            (default 2, the Euclidean/L2 norm). `1` gives the L1 norm;
            `np.inf` gives the max-absolute-value norm.

    Returns:
        The gradient norm as a float. May be `nan` or `inf` if the input
        contains non-finite values -- that is meaningful signal (a
        diverging gradient), not an error condition here.

    Raises:
        ValueError: If `gradients` is empty.
    """
    flat = _flatten_and_validate(gradients, "gradient_norm")
    return float(np.linalg.norm(flat, ord=ord))


def mean_absolute_gradient(gradients) -> float:
    """Compute the mean of the absolute gradient values.

    Args:
        gradients: Array-like of gradient values, any shape.

    Returns:
        Mean absolute gradient value as a float.

    Raises:
        ValueError: If `gradients` is empty.
    """
    flat = _flatten_and_validate(gradients, "mean_absolute_gradient")
    return float(np.mean(np.abs(flat)))


def gradient_variance(gradients) -> float:
    """Compute the (population) variance of the signed gradient values.

    Uses `ddof=0` (population variance, dividing by N), matching
    `schemas.gradient.summarize_gradient`'s `variance` field so the two
    stay consistent for callers that use both.

    Args:
        gradients: Array-like of gradient values, any shape.

    Returns:
        Variance of the flattened gradient values as a float.

    Raises:
        ValueError: If `gradients` is empty.
    """
    flat = _flatten_and_validate(gradients, "gradient_variance")
    return float(np.var(flat))


def gradient_percentiles(
    gradients,
    percentiles: tuple[float, ...] = (1, 10, 50, 90, 99),
) -> dict[float, float]:
    """Compute percentiles of the (signed) gradient distribution.

    Args:
        gradients: Array-like of gradient values, any shape.
        percentiles: Percentile ranks to compute, each in `[0, 100]`.
            Defaults to a spread useful for spotting a heavy-tailed or
            collapsed gradient distribution.

    Returns:
        A dict mapping each requested percentile rank to its value, e.g.
        `{1: -0.02, 10: -0.01, 50: 0.0, 90: 0.01, 99: 0.02}`.

    Raises:
        ValueError: If `gradients` is empty, `percentiles` is empty, or
            any requested percentile is outside `[0, 100]`.
    """
    flat = _flatten_and_validate(gradients, "gradient_percentiles")
    if len(percentiles) == 0:
        raise ValueError("gradient_percentiles() requires at least one percentile rank")
    for p in percentiles:
        if not (0 <= p <= 100):
            raise ValueError(f"percentile rank must be in [0, 100], got {p}")

    values = np.percentile(flat, list(percentiles))
    return {p: float(v) for p, v in zip(percentiles, values, strict=True)}
