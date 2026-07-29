"""Confidence intervals for gradient statistics.

Milestone 9 (Volume IV, `statistics/confidence.py`), Issue #69.

Two ways to put an uncertainty band around an estimated gradient L2 norm:

- `estimate_gradient_norm_ci`: a cheap, O(1) analytic interval (delta
  method), safe to compute on every step. This is the one detectors use
  by default (`attach_gradient_norm_ci` below, wired into
  `BarrenPlateauDetector`) -- plan.md §26 is explicit that the default
  monitoring path must stay cheap enough to run continuously, so a
  per-step confidence interval cannot mean "run a bootstrap every step".
- `bootstrap_gradient_norm_ci`: a heavier percentile-bootstrap
  alternative, useful for offline/exploratory analysis of a single
  gradient snapshot's raw values, but deliberately *not* called
  automatically by any detector.

Delta-method derivation
------------------------
Model a gradient vector as `g = mu + eps`, where `mu` is the (unknown)
true gradient and `eps` is homoscedastic per-component noise with
variance `se**2` (independent across components). A first-order Taylor
expansion of the norm around `mu` gives:

    ||g|| ~= ||mu|| + (mu . eps) / ||mu||
    Var(||g||) ~= se**2 * ||mu||**2 / ||mu||**2 = se**2

i.e. the norm's own standard error is approximately `se` itself,
*independent of the number of parameters* -- this holds whenever `||mu||`
is not tiny relative to `se` (the approximation degrades exactly in the
near-collapse regime, which is precisely where the interval matters most
for a barren-plateau report; this is documented as a known limitation
rather than glossed over, see `attach_gradient_norm_ci`).

What `se` should be is context-dependent, which is why this module
exposes the interval computation and `attach_gradient_norm_ci` (below)
separately from the choice of `se`:

- If `shots` is known: `se = estimate_measurement_uncertainty(variance,
  shots)` (`statistics.snr`) -- the interval reflects genuine shot-noise
  uncertainty on the norm.
- If `shots` is unknown (analytic/adjoint execution): `se =
  sqrt(variance)`, where `variance` is the gradient's own
  *across-parameter* spread (`GradientSnapshot.variance`). This is a
  different, weaker uncertainty statement -- "how much would this norm
  plausibly differ if estimated from a different, similarly-distributed
  set of circuit parameters" -- not a measurement-noise claim, and is
  labeled as such (`ci_method="parameter-spread-analytic"`) rather than
  presented as if it meant the same thing as the shot-noise case. This
  mirrors the same care taken in `detectors/noise.py` not to conflate
  parameter-spread with measurement noise.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from qml_observer.schemas.gradient import GradientSnapshot
from qml_observer.statistics.snr import estimate_measurement_uncertainty


def _inverse_normal_cdf(p: float) -> float:
    """Standard normal quantile function (probit).

    Peter Acklam's rational approximation -- pure Python/math, no scipy
    dependency (this project's only runtime dependency is numpy; see
    `pyproject.toml`). Accurate to within ~1.15e-9 relative error over
    `(0, 1)`, far more precision than a confidence-interval half-width
    needs.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def _z_score(confidence: float) -> float:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    tail = (1.0 - confidence) / 2.0
    return _inverse_normal_cdf(1.0 - tail)


def estimate_gradient_norm_ci(
    norm_l2: float, se: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Analytic (delta-method) confidence interval on a gradient L2 norm.

    Args:
        norm_l2: The observed/estimated gradient L2 norm.
        se: Standard error of that norm estimate (see module docstring
            for how to derive one). Must be `>= 0` (or NaN).
        confidence: Confidence level, in `(0, 1)`. Default `0.95`.

    Returns:
        `(lower, upper)`, with `lower` clamped to `0.0` (a norm can never
        be negative). `(nan, nan)` if `norm_l2` or `se` is NaN.

    Raises:
        ValueError: If `se` is a finite negative number, or `confidence`
            is not in `(0, 1)`.
    """
    if math.isnan(se):
        return (float("nan"), float("nan"))
    if se < 0:
        raise ValueError(f"se must be >= 0 (or NaN), got {se}")
    if math.isnan(norm_l2):
        return (float("nan"), float("nan"))

    z = _z_score(confidence)
    half_width = z * se

    if math.isinf(norm_l2) or math.isinf(half_width):
        lower = 0.0 if norm_l2 >= 0 else float("-inf")
        return (max(0.0, lower), float(norm_l2 + half_width))

    lower = max(0.0, norm_l2 - half_width)
    upper = norm_l2 + half_width
    return (lower, upper)


def bootstrap_gradient_norm_ci(
    gradient_values: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval on a gradient's L2 norm.

    Resamples the gradient's *components* with replacement `n_resamples`
    times (treating the observed parameters as an exchangeable sample from
    some underlying per-parameter gradient distribution), recomputes the
    L2 norm of each resample, and returns the empirical percentile
    interval. This is a different uncertainty statement from
    `estimate_gradient_norm_ci`'s shot-noise/measurement interval: it
    reflects parameter-to-parameter variability, not measurement
    precision, and needs the raw per-parameter values (not just summary
    statistics) to compute.

    Deliberately not called by any detector's per-step `update()`/
    `diagnose()` path (plan.md §26: "avoid heavy computations on every
    step... make all advanced diagnostics optional"); intended for
    offline/notebook use on a specific step of interest.

    Args:
        gradient_values: The raw gradient array for one step (any shape;
            flattened internally). Must be non-empty.
        confidence: Confidence level, in `(0, 1)`. Default `0.95`.
        n_resamples: Number of bootstrap resamples.
        seed: Optional seed for reproducibility.

    Returns:
        `(lower, upper)` percentile interval on the L2 norm.

    Raises:
        ValueError: If `gradient_values` is empty, `n_resamples < 1`, or
            `confidence` is not in `(0, 1)`.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    flat = np.asarray(gradient_values, dtype=float).ravel()
    if flat.size == 0:
        raise ValueError("bootstrap_gradient_norm_ci() received an empty gradient array.")

    rng = np.random.default_rng(seed)
    n = flat.size
    resample_indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_norms = np.linalg.norm(flat[resample_indices], ord=2, axis=1)

    tail = (1.0 - confidence) / 2.0
    lower = float(np.percentile(resampled_norms, 100 * tail))
    upper = float(np.percentile(resampled_norms, 100 * (1.0 - tail)))
    return (max(0.0, lower), upper)


def attach_gradient_norm_ci(
    snapshot: GradientSnapshot,
    *,
    shots: int | None = None,
    confidence: float = 0.95,
) -> GradientSnapshot:
    """Return a copy of `snapshot` with an analytic norm CI attached.

    Chooses the standard error automatically per this module's docstring:
    shot-noise-derived when `shots` is a positive int, else derived from
    the snapshot's own across-parameter `variance`. O(1) either way --
    safe to call on every step (this is what `BarrenPlateauDetector` does
    to surface uncertainty in its evidence, Issue #69).

    Args:
        snapshot: The `GradientSnapshot` to attach an interval to. Any
            existing `ci_*` fields on it are overwritten.
        shots: Shot count for this step, if known (e.g.
            `StepObservation.shots`). `None` or `<= 0` falls back to the
            parameter-spread method.
        confidence: Confidence level, in `(0, 1)`. Default `0.95`.

    Returns:
        A new `GradientSnapshot` (the input is not mutated) with
        `ci_lower`, `ci_upper`, `ci_level`, and `ci_method` populated.
    """
    if shots is not None and shots > 0:
        se = estimate_measurement_uncertainty(snapshot.variance, shots)
        method = "shot-noise-analytic"
    else:
        variance = snapshot.variance
        se = float("nan") if math.isnan(variance) else math.sqrt(max(0.0, variance))
        method = "parameter-spread-analytic"

    ci_lower, ci_upper = estimate_gradient_norm_ci(snapshot.norm_l2, se, confidence=confidence)

    return replace(
        snapshot,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=confidence,
        ci_method=method,
    )
