"""Quantum Fisher Information Matrix (QFIM) estimation.

Milestone 12 (blueprint Volume XIII), Issue #83.

## Mathematical description

For a parameterized pure state `|psi(theta)>` produced by a variational
circuit, the QFIM is the (real part of the) Fubini-Study metric tensor:

    F_ij(theta) = 4 * Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]

where `d_i psi = d|psi(theta)>/d(theta_i)`. It is the natural Riemannian
metric on the manifold of quantum states reachable by the circuit, and is
the object natural-gradient descent preconditions the ordinary gradient
with (`F^-1 grad`, out of scope here -- see `RecoveryPlanner`,
Milestone 13). A near-singular QFIM at the current parameters means many
directions in parameter space barely change the output state at all:
exactly the local geometric signature "barren plateau" theory predicts,
and the reason this module exists as a *complement* to the gradient-based
`BarrenPlateauDetector` (Milestone 4) rather than a replacement for it --
a small gradient says the *loss* is flat; a near-singular QFIM says the
*circuit's expressivity at this point* is flat, which is a different (if
related) statement.

## Estimation method

Per the blueprint's adapter philosophy ("observe the result, don't
reimplement the framework's differentiation machinery"), this module does
not depend on PennyLane, Qiskit, or any autodiff framework. It treats the
circuit purely as a black-box function `parameters -> statevector`
(`state_fn`, a plain Python callable returning a 1D complex `numpy`
array), which the caller constructs however is convenient in their
framework (e.g. `lambda p: qml.numpy.array(circuit(p))` in PennyLane, or
`Statevector(qc.assign_parameters(p)).data` in Qiskit).

The metric tensor's derivatives `d_i psi` are estimated by central finite
differences of `state_fn` (`(state_fn(theta + eps*e_i) - state_fn(theta -
eps*e_i)) / (2*eps)`), which is the standard, framework-agnostic
fallback method when an analytic parameter-shift QFIM rule for the
specific ansatz is not being hand-derived (block-diagonal analytic QFIM
methods, e.g. Stokes et al. 2020, are a documented future extension --
see Known Limitations below and `docs/research/geometry.md`).

## Cost

`O(n_parameters)` calls to `state_fn` for the derivative estimates, plus
`O(n_parameters^2)` vectorized inner products to assemble the matrix --
this is a research/diagnostic-mode-only cost (plan.md §26), never called
from the per-step monitoring path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def estimate_qfim(
    state_fn: Callable[[np.ndarray], np.ndarray],
    parameters: Sequence[float] | np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Estimate the quantum Fisher information matrix at `parameters`.

    Args:
        state_fn: Callable mapping a 1D real parameter array to the
            circuit's output statevector, as a 1D complex (or real)
            `numpy` array of fixed length (`2**n_qubits`). Must be
            deterministic (e.g. an exact/analytic statevector simulation,
            not a finite-shots sampled estimate -- finite differences of
            a noisy function amplify the noise; see Known Limitations in
            `docs/research/geometry.md`).
        parameters: The 1D parameter vector `theta` to evaluate the QFIM
            at.
        eps: Finite-difference step size for estimating `d_i psi`. The
            default (`1e-4`) balances truncation error (large `eps`)
            against floating-point cancellation error (tiny `eps`) for
            typical circuit outputs in `float64`/`complex128`; tune it
            down for very smooth `state_fn`s and up if the result looks
            noisy.

    Returns:
        The `(n_parameters, n_parameters)` real, symmetric, positive
        semi-definite QFIM as a `numpy` array. Symmetry is enforced
        exactly (`0.5 * (F + F.T)`) to cancel finite-difference asymmetry
        that would otherwise appear in the strictly-mathematical result
        due to floating-point rounding, not due to any asymmetry in the
        true QFIM itself (which is always symmetric).

    Raises:
        ValueError: If `parameters` is empty, or if `state_fn` returns a
            state whose norm is not finite and positive (e.g. all-zero,
            NaN, or Inf -- almost always a caller bug in `state_fn`
            rather than a legitimate training signal, unlike loss/
            gradient NaNs, which is why this raises rather than
            propagating per addendum §7's numerical-edge-case
            convention).
    """
    theta = np.asarray(parameters, dtype=float)
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError(f"parameters must be a non-empty 1D array, got shape {theta.shape}")

    psi = np.asarray(state_fn(theta), dtype=complex)
    norm = np.linalg.norm(psi)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            "state_fn(parameters) must return a state with finite, positive norm; "
            f"got norm={norm!r}"
        )
    psi = psi / norm

    n = theta.size
    derivatives = np.empty((n, psi.size), dtype=complex)
    for i in range(n):
        shift = np.zeros(n, dtype=float)
        shift[i] = eps
        psi_plus = np.asarray(state_fn(theta + shift), dtype=complex)
        psi_minus = np.asarray(state_fn(theta - shift), dtype=complex)
        psi_plus = psi_plus / np.linalg.norm(psi_plus)
        psi_minus = psi_minus / np.linalg.norm(psi_minus)
        derivatives[i] = (psi_plus - psi_minus) / (2.0 * eps)

    # <d_i psi | d_j psi>
    overlap = derivatives.conj() @ derivatives.T
    # <d_i psi | psi> and <psi | d_j psi>
    proj = derivatives.conj() @ psi
    fim = 4.0 * np.real(overlap - np.outer(proj, proj.conj()))
    return 0.5 * (fim + fim.T)
