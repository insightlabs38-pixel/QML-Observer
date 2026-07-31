"""Finite-difference Hessian-vector product (HVP) estimation.

Milestone 12 (blueprint Volume XIII), Issue #86.

## Mathematical description

For a scalar loss `f(theta)`, the Hessian-vector product `H(theta) @ v`
(where `H` is the `n x n` Hessian of `f` at `theta`) captures the
curvature of the loss along direction `v`, without needing the full
`O(n^2)`-sized Hessian matrix. It is the observation this module exists
to provide -- per blueprint Volume XIII, purely for *inspection*
(e.g. characterizing curvature near a suspected plateau), not to power an
optimizer step; natural-gradient/second-order optimization is explicitly
out of scope here and belongs to `RecoveryPlanner` (Milestone 13) if ever
built.

## Estimation method: double finite difference, loss-only

Per the blueprint's exact signature (`estimate_hessian_vector_product(
loss_fn, parameters, vector)`) and its "observe the result, don't
reimplement the framework's differentiation machinery" philosophy, this
estimates `Hv` from `loss_fn` alone -- no analytic gradient or Hessian
function is required from the caller. It composes two central-difference
gradient estimates:

    grad(theta) ~= central finite difference of loss_fn, per-parameter
    Hv ~= (grad(theta + h*v) - grad(theta - h*v)) / (2*h)

i.e. a finite difference of finite differences. This is the standard
black-box fallback for HVP estimation when only zeroth-order (function
value) access is available (cf. Pearlmutter's exact HVP trick, which
requires access to an analytic/autodiff gradient function and is *not*
used here for exactly that reason -- it would mean reimplementing or
depending on the framework's own differentiation machinery, which the
blueprint's adapter philosophy explicitly avoids).

## Cost and accuracy trade-off

Each call costs `4 * n_parameters` evaluations of `loss_fn` (two
`n_parameters`-sized central-difference gradients). This is markedly more
expensive than the `O(n_parameters)` cost of `qfim.estimate_qfim`, and is
documented as a research/diagnostic-only cost, never appropriate in the
per-step monitoring path (plan.md §26). Because it nests two finite
differences, floating-point cancellation error compounds; the default
step sizes below are chosen as a reasonable default for `float64` loss
values of order `O(1)` and should be tuned (larger `h`/`grad_eps`) for
very noisy or very small-magnitude losses -- see Known Limitations in
`docs/research/geometry.md`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _finite_difference_gradient(
    loss_fn: Callable[[np.ndarray], float], parameters: np.ndarray, grad_eps: float
) -> np.ndarray:
    grad = np.empty_like(parameters, dtype=float)
    for i in range(parameters.size):
        shift = np.zeros_like(parameters)
        shift[i] = grad_eps
        grad[i] = (loss_fn(parameters + shift) - loss_fn(parameters - shift)) / (2.0 * grad_eps)
    return grad


def estimate_hessian_vector_product(
    loss_fn: Callable[[np.ndarray], float],
    parameters: Sequence[float] | np.ndarray,
    vector: Sequence[float] | np.ndarray,
    eps: float = 1e-4,
    grad_eps: float = 1e-4,
) -> np.ndarray:
    """Estimate `H(parameters) @ vector` via nested finite differences.

    Args:
        loss_fn: Callable mapping a 1D real parameter array to a scalar
            loss. Should be deterministic/low-noise for a meaningful
            result (see module docstring on cancellation error); a
            finite-shots-sampled loss will produce a very noisy HVP
            estimate.
        parameters: The 1D parameter vector `theta` to evaluate the HVP
            at.
        vector: The 1D direction vector `v` to contract the Hessian
            against. Must be the same length as `parameters`.
        eps: Outer finite-difference step size, scaling how far along
            `vector` the two gradient evaluations are taken.
        grad_eps: Inner finite-difference step size used for each
            per-parameter gradient estimate.

    Returns:
        A 1D `numpy` array of the same length as `parameters`,
        approximating `H @ vector`.

    Raises:
        ValueError: If `parameters` and `vector` are not both non-empty
            1D arrays of equal length, or if `eps`/`grad_eps` are not
            strictly positive.
    """
    theta = np.asarray(parameters, dtype=float)
    v = np.asarray(vector, dtype=float)
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError(f"parameters must be a non-empty 1D array, got shape {theta.shape}")
    if v.shape != theta.shape:
        raise ValueError(
            f"vector must have the same shape as parameters, got {v.shape} vs {theta.shape}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if grad_eps <= 0:
        raise ValueError(f"grad_eps must be > 0, got {grad_eps}")

    grad_plus = _finite_difference_gradient(loss_fn, theta + eps * v, grad_eps)
    grad_minus = _finite_difference_gradient(loss_fn, theta - eps * v, grad_eps)
    return (grad_plus - grad_minus) / (2.0 * eps)
