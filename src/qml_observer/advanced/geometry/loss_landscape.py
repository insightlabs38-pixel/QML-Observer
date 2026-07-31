"""Loss-landscape sampling utilities.

Milestone 12 (blueprint Volume XIII), Issue #87.

## Purpose

`estimate_qfim`/`estimate_hessian_vector_product` characterize the local
geometry at a single point analytically (to first/second order). This
module instead directly samples `loss_fn` along one or two directions
through parameter space, producing an explicit curve/surface a user can
plot -- useful for sanity-checking the local analytic diagnostics above
against ground truth (e.g. "does the landscape actually look flat where
the QFIM says it's near-singular?") and for the qualitative "is this loss
landscape suspiciously flat everywhere nearby" question that a single
gradient/Hessian sample can't answer on its own.

No statistical/detection logic lives here (per the blueprint's detection/
diagnosis separation, applied here too): these functions return raw
sampled values. `landscape_flatness` is a plain descriptive summary
(range and standard deviation of the sampled losses), not a verdict.

## Cost

`sample_loss_landscape_1d` costs `n_points` evaluations of `loss_fn`;
`sample_loss_landscape_2d` costs `n_points ** 2`. Both are
research/diagnostic-only, not part of the per-step monitoring path
(plan.md §26) -- a 2D scan with the default `n_points=21` already costs
441 loss evaluations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class LandscapeSample:
    """A sampled loss landscape along one or two directions.

    Attributes:
        alphas: 1D array of length `n_points` -- the coefficients sampled
            along `direction` (1D scans) or `direction1` (2D scans, first
            axis).
        betas: 1D array of length `n_points` sampled along `direction2`
            for 2D scans, or `None` for 1D scans.
        losses: For a 1D scan, a 1D array of length `n_points` (loss at
            each `alphas[i]`). For a 2D scan, a 2D array of shape
            `(n_points, n_points)` where `losses[i, j]` is the loss at
            `parameters + alphas[i] * direction1 + betas[j] * direction2`.
    """

    alphas: np.ndarray
    losses: np.ndarray
    betas: np.ndarray | None = None


def random_direction(n_parameters: int, seed: int | None = None) -> np.ndarray:
    """Generate a random unit-norm direction in parameter space.

    Args:
        n_parameters: Dimensionality of the direction vector.
        seed: Optional seed for reproducibility -- callers comparing
            landscapes across runs/circuits should fix this so the same
            random direction (in coefficient terms) is used for both,
            since an unseeded random direction makes cross-run
            comparisons meaningless.

    Returns:
        A 1D `numpy` array of length `n_parameters` with unit L2 norm,
        drawn from an isotropic standard normal distribution (Gaussian
        components then normalized -- the standard way to sample
        uniformly from the unit sphere).

    Raises:
        ValueError: If `n_parameters <= 0`.
    """
    if n_parameters <= 0:
        raise ValueError(f"n_parameters must be > 0, got {n_parameters}")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(n_parameters)
    norm = np.linalg.norm(vec)
    return vec / norm


def sample_loss_landscape_1d(
    loss_fn: Callable[[np.ndarray], float],
    parameters: Sequence[float] | np.ndarray,
    direction: Sequence[float] | np.ndarray,
    span: tuple[float, float] = (-1.0, 1.0),
    n_points: int = 21,
) -> LandscapeSample:
    """Sample `loss_fn` along a single direction through `parameters`.

    Evaluates `loss_fn(parameters + alpha * direction)` for `n_points`
    evenly spaced `alpha` values in `span`.

    Args:
        loss_fn: Callable mapping a 1D real parameter array to a scalar
            loss.
        parameters: The center point `theta` to scan around.
        direction: The direction vector to scan along. Need not be
            unit-norm (e.g. pass `random_direction(...)` for a
            normalized scan, or an unnormalized direction of interest
            such as a specific QFIM eigenvector).
        span: `(alpha_min, alpha_max)` range to sample over.
        n_points: Number of evenly spaced sample points, inclusive of
            both endpoints of `span`. Must be `>= 2`.

    Returns:
        A `LandscapeSample` with `betas=None`.

    Raises:
        ValueError: If `parameters`/`direction` are not equal-length,
            non-empty 1D arrays, if `n_points < 2`, or if
            `span[0] >= span[1]`.
    """
    theta = np.asarray(parameters, dtype=float)
    d = np.asarray(direction, dtype=float)
    _validate_point_and_direction(theta, d)
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if span[0] >= span[1]:
        raise ValueError(f"span must satisfy span[0] < span[1], got {span}")

    alphas = np.linspace(span[0], span[1], n_points)
    losses = np.array([loss_fn(theta + alpha * d) for alpha in alphas])
    return LandscapeSample(alphas=alphas, losses=losses)


def sample_loss_landscape_2d(
    loss_fn: Callable[[np.ndarray], float],
    parameters: Sequence[float] | np.ndarray,
    direction1: Sequence[float] | np.ndarray,
    direction2: Sequence[float] | np.ndarray,
    span: tuple[float, float] = (-1.0, 1.0),
    n_points: int = 21,
) -> LandscapeSample:
    """Sample `loss_fn` over a 2D grid spanned by two directions.

    Evaluates `loss_fn(parameters + alpha * direction1 + beta *
    direction2)` over an `n_points x n_points` grid of `(alpha, beta)`
    pairs, both ranging over `span`.

    Args, Returns, Raises: as `sample_loss_landscape_1d`, plus
    `direction2` (validated the same way as `direction1`). `betas` in the
    returned `LandscapeSample` is populated (equal to `alphas` since both
    axes share `span`/`n_points` by design, for a square grid suitable
    for a contour/surface plot).
    """
    theta = np.asarray(parameters, dtype=float)
    d1 = np.asarray(direction1, dtype=float)
    d2 = np.asarray(direction2, dtype=float)
    _validate_point_and_direction(theta, d1)
    _validate_point_and_direction(theta, d2)
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if span[0] >= span[1]:
        raise ValueError(f"span must satisfy span[0] < span[1], got {span}")

    alphas = np.linspace(span[0], span[1], n_points)
    betas = np.linspace(span[0], span[1], n_points)
    losses = np.empty((n_points, n_points), dtype=float)
    for i, alpha in enumerate(alphas):
        base = theta + alpha * d1
        for j, beta in enumerate(betas):
            losses[i, j] = loss_fn(base + beta * d2)
    return LandscapeSample(alphas=alphas, betas=betas, losses=losses)


def landscape_flatness(sample: LandscapeSample) -> dict[str, float]:
    """Summarize how flat a sampled landscape is.

    A plain descriptive summary, not a verdict (see module docstring):
    callers/future detectors decide what a "flat" value means for their
    purposes.

    Args:
        sample: A `LandscapeSample` from `sample_loss_landscape_1d` or
            `sample_loss_landscape_2d`.

    Returns:
        A dict with `"range"` (`max(losses) - min(losses)`), `"std"`
        (population standard deviation of the sampled losses), and
        `"mean"`. `nan` propagates into all three if any sampled loss is
        `nan` (consistent with addendum §7: a diverging/unstable loss is
        meaningful signal, not something to silently mask here).
    """
    losses = sample.losses
    return {
        "range": float(np.max(losses) - np.min(losses)),
        "std": float(np.std(losses)),
        "mean": float(np.mean(losses)),
    }


def _validate_point_and_direction(theta: np.ndarray, direction: np.ndarray) -> None:
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError(f"parameters must be a non-empty 1D array, got shape {theta.shape}")
    if direction.shape != theta.shape:
        raise ValueError(
            f"direction must have the same shape as parameters, got "
            f"{direction.shape} vs {theta.shape}"
        )
