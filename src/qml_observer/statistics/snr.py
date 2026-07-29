"""Gradient SNR and shot-noise uncertainty primitives.

Milestone 9 (Volume IV, `statistics/snr.py`), Issues #64-#65:

- Issue #64: `estimate_gradient_snr`
- Issue #65: `estimate_measurement_uncertainty`

Both functions were scaffolded (signature only) in the blueprint and are
implemented here for the first time. They are the statistical primitives
`NoiseDetector` (Issue #66) is built on: distinguishing "the gradient is
genuinely near zero" (a real barren plateau, `BarrenPlateauDetector`'s
concern) from "the gradient *estimate* is too noisy/under-sampled to draw
any conclusion from yet" (this module's concern) is the central false-
positive-reduction goal of Milestone 9 (plan.md §21, addendum §3).

Numerical edge cases (addendum §7, extended here per the same bar applied
to every other statistics module): zero standard deviation, zero variance,
and NaN/Inf inputs are all handled explicitly and documented below, never
left to raise an opaque `ZeroDivisionError` or numpy `RuntimeWarning`.
"""

from __future__ import annotations

import math


def _check_finite_or_nan(value: float, name: str) -> float:
    """Coerce to float, rejecting non-numeric input; NaN passes through.

    Mirrors the project-wide convention (`statistics.loss`, `schemas.
    _validation.check_non_negative_number`): a non-finite *value* is
    meaningful signal from an unstable/degenerate run and must never be
    rejected, but the *type* must be a real number.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value)!r}")
    return float(value)


def estimate_gradient_snr(mean_gradient: float, gradient_std: float) -> float:
    """Estimate a gradient's signal-to-noise ratio.

    Defined as ``abs(mean_gradient) / gradient_std``: how large the
    gradient's estimated magnitude is relative to its own spread. A high
    value means the gradient estimate is trustworthy (the signal clearly
    exceeds the noise floor); a low value means the estimate could easily
    be noise -- it says nothing, on its own, about whether the *true*
    underlying gradient is large or small.

    `mean_gradient` is taken as a magnitude via `abs()` internally, so
    callers may pass either a signed mean (e.g. the mean of a single
    gradient component across repeated measurements) or an already-
    unsigned quantity (e.g. `GradientSnapshot.mean_abs`) without changing
    the result's sign convention.

    Args:
        mean_gradient: The estimated mean gradient value (signed or
            unsigned; sign is discarded).
        gradient_std: The estimated standard deviation of that same
            gradient estimate. Must be `>= 0` (or NaN).

    Returns:
        The SNR as a non-negative float, or:
          - `nan` if `gradient_std` is NaN (an undefined spread is an
            undefined ratio, not a zero one).
          - `0.0` if `gradient_std == 0.0` *and* the gradient magnitude is
            also exactly `0.0` -- a fully degenerate all-zero estimate.
            This is deliberately reported as "no signal" rather than
            "infinite confidence": zero noise and zero signal are
            indistinguishable from "nothing was measured", and claiming
            perfect SNR here would be misleading for a detector deciding
            whether to trust the estimate.
          - `inf` if `gradient_std == 0.0` and the gradient magnitude is
            nonzero -- a perfectly noiseless, clearly nonzero estimate
            (e.g. an analytic, infinite-shot gradient).
          - `nan` if `mean_gradient` is NaN (propagates; an undefined mean
            makes the ratio undefined too).

    Raises:
        TypeError: If either argument is not a real number.
        ValueError: If `gradient_std` is a finite negative number (a
            standard deviation can never legitimately be negative).
    """
    mean_gradient = _check_finite_or_nan(mean_gradient, "mean_gradient")
    gradient_std = _check_finite_or_nan(gradient_std, "gradient_std")

    if math.isnan(gradient_std):
        return float("nan")
    if gradient_std < 0:
        raise ValueError(f"gradient_std must be >= 0 (or NaN), got {gradient_std}")

    if math.isnan(mean_gradient):
        return float("nan")

    magnitude = abs(mean_gradient)

    if gradient_std == 0.0:
        return 0.0 if magnitude == 0.0 else float("inf")

    return magnitude / gradient_std


def estimate_measurement_uncertainty(expectation_variance: float, shots: int) -> float:
    """Estimate the shot-noise uncertainty of an expectation value.

    Uses the standard-error-of-the-mean formula for an expectation value
    estimated by averaging `shots` independent measurement outcomes:

        uncertainty = sqrt(expectation_variance / shots)

    where `expectation_variance` is the per-shot variance of the
    underlying measurement outcome (e.g. for a Pauli expectation value in
    {-1, +1}, `expectation_variance <= 1`). This is the finite-shots
    "noise floor" that `NoiseDetector` (Issue #66) compares an observed
    gradient magnitude against: a gradient that looks collapsed but whose
    magnitude is smaller than this uncertainty cannot yet be distinguished
    from shot noise, regardless of how small it appears.

    Args:
        expectation_variance: Per-shot variance of the measured
            expectation value. Must be `>= 0` (or NaN); variance can
            never legitimately be negative, unlike loss/gradient values
            which may be.
        shots: Number of shots (measurements) the expectation value was
            estimated from. Must be a positive int -- `shots=None`
            (analytic/infinite-shots execution) has no shot-noise floor
            to estimate and callers should not call this function in
            that case at all, rather than passing a placeholder.

    Returns:
        The estimated uncertainty as a non-negative float. `nan` if
        `expectation_variance` is NaN (propagates). `inf` if
        `expectation_variance` is `inf`.

    Raises:
        TypeError: If `shots` is not an int, or `expectation_variance` is
            not a real number.
        ValueError: If `shots <= 0`, or `expectation_variance` is a finite
            negative number.
    """
    if not isinstance(shots, int) or isinstance(shots, bool):
        raise TypeError(f"shots must be an int, got {type(shots)!r}")
    if shots <= 0:
        raise ValueError(f"shots must be > 0, got {shots}")

    expectation_variance = _check_finite_or_nan(expectation_variance, "expectation_variance")

    if math.isnan(expectation_variance):
        return float("nan")
    if expectation_variance < 0:
        raise ValueError(
            f"expectation_variance must be >= 0 (or NaN), got {expectation_variance}"
        )
    if math.isinf(expectation_variance):
        return float("inf")

    return math.sqrt(expectation_variance / shots)
