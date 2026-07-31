"""QFIM conditioning: condition number and effective rank.

Milestone 12 (blueprint Volume XIII), Issue #84, built on `qfim.py`
(Issue #83).

## Mathematical description

Given the QFIM's eigenvalues `lambda_1 >= lambda_2 >= ... >= lambda_n >= 0`
(real and non-negative since the QFIM is symmetric PSD):

- **Condition number**: `kappa = lambda_1 / lambda_n`. A large condition
  number means the local optimization landscape is extremely
  anisotropic -- some parameter directions change the state's
  distinguishability enormously per unit step, others barely at all --
  which is exactly the situation vanilla gradient descent struggles with
  and natural-gradient methods (`F^-1 grad`, Milestone 13) are designed
  to correct for. `n_zero` near-zero eigenvalues make `kappa` numerically
  `inf`; see `rcond` below for how that is handled without raising.
- **Effective rank**: the number of eigenvalues that are "meaningfully
  nonzero" relative to `lambda_1`, i.e. a soft rank rather than an exact
  linear-algebra rank (which would be brittle for a QFIM whose eigenvalues
  form a continuum near zero rather than being exactly zero). This module
  uses the simple, explainable threshold definition
  `effective_rank = count(lambda_i >= threshold * lambda_1)`, matching
  the blueprint's stated goal of a first, deterministic, explainable
  scoring approach (per Volume VII's diagnosis-engine philosophy, applied
  here to a single-metric scoring choice) -- more sophisticated entropy-
  based effective-rank definitions (e.g. Roy & Vetterli 2007) are a
  documented possible future extension in `docs/research/geometry.md`,
  not implemented here to keep the first version auditable by inspection.

## Interpretation, not diagnosis

Per the blueprint's second architectural rule (detection vs. diagnosis
separation, applied here by analogy): this module reports numbers, not
verdicts. `qfim_condition_number`/`effective_rank`/`summarize_conditioning`
never themselves conclude "this circuit has a barren plateau" -- that
interpretation, if built into a detector at all, is future work
(`docs/research/geometry.md`, Known Limitations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConditioningResult:
    """Summary of a QFIM's spectral conditioning at one parameter point.

    Attributes:
        eigenvalues: The QFIM's eigenvalues, descending
            (`eigenvalues[0]` is the largest).
        condition_number: `eigenvalues[0] / eigenvalues[-1]`, or `inf`
            if the smallest eigenvalue is (numerically) zero.
        effective_rank: See `effective_rank()` below.
        n_parameters: The QFIM's dimension (`len(eigenvalues)`).
    """

    eigenvalues: np.ndarray
    condition_number: float
    effective_rank: int
    n_parameters: int


def _eigenvalues_descending(qfim: np.ndarray) -> np.ndarray:
    if qfim.ndim != 2 or qfim.shape[0] != qfim.shape[1]:
        raise ValueError(f"qfim must be a square 2D array, got shape {qfim.shape}")
    if qfim.shape[0] == 0:
        raise ValueError("qfim must be non-empty")
    # eigvalsh assumes/expects a symmetric (Hermitian) matrix, which the
    # QFIM always is by construction -- using it (rather than eig) also
    # guarantees real eigenvalues without needing a defensive cast.
    eigenvalues = np.linalg.eigvalsh(qfim)
    # Numerically-negative-but-should-be-zero eigenvalues can appear from
    # finite-difference/floating-point error even though the QFIM is
    # mathematically PSD; clip rather than let a spurious tiny negative
    # value silently break condition_number's ordering assumption.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return eigenvalues[::-1]


def qfim_condition_number(qfim: np.ndarray, rcond: float = 1e-12) -> float:
    """Compute the QFIM's condition number `lambda_max / lambda_min`.

    Args:
        qfim: A square, symmetric QFIM (e.g. from `estimate_qfim`).
        rcond: Relative threshold, as a fraction of `lambda_max`, below
            which the smallest eigenvalue is treated as exactly zero
            (yielding `condition_number = inf`) rather than dividing by a
            near-zero floating-point value and returning a large but
            finite, essentially arbitrary number.

    Returns:
        The condition number as a non-negative float, or `inf` if
        `lambda_max <= 0` (an all-zero QFIM -- every direction is
        equally, completely uninformative) or if the smallest eigenvalue
        falls below `rcond * lambda_max`.
    """
    eigenvalues = _eigenvalues_descending(qfim)
    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[-1]
    if lambda_max <= 0:
        return float("inf")
    if lambda_min <= rcond * lambda_max:
        return float("inf")
    return float(lambda_max / lambda_min)


def effective_rank(qfim: np.ndarray, threshold: float = 1e-6) -> int:
    """Count eigenvalues that are "meaningfully nonzero" relative to the largest.

    Args:
        qfim: A square, symmetric QFIM (e.g. from `estimate_qfim`).
        threshold: Relative cutoff, as a fraction of `lambda_max`. An
            eigenvalue `lambda_i` counts toward the effective rank iff
            `lambda_i >= threshold * lambda_max`.

    Returns:
        An integer in `[0, n_parameters]`. `0` iff every eigenvalue is
        `<= 0` (a fully degenerate, all-zero QFIM).
    """
    eigenvalues = _eigenvalues_descending(qfim)
    lambda_max = eigenvalues[0]
    if lambda_max <= 0:
        return 0
    return int(np.sum(eigenvalues >= threshold * lambda_max))


def summarize_conditioning(
    qfim: np.ndarray, rcond: float = 1e-12, threshold: float = 1e-6
) -> ConditioningResult:
    """Compute `ConditioningResult` for `qfim` in one call.

    Convenience wrapper around `qfim_condition_number` and
    `effective_rank` that eigendecomposes `qfim` only once (both
    functions above are also public/independently usable, at the cost of
    a redundant eigendecomposition if both are called separately on a
    large QFIM).
    """
    eigenvalues = _eigenvalues_descending(qfim)
    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[-1]
    if lambda_max <= 0:
        condition_number = float("inf")
    elif lambda_min <= rcond * lambda_max:
        condition_number = float("inf")
    else:
        condition_number = float(lambda_max / lambda_min)
    rank = 0 if lambda_max <= 0 else int(np.sum(eigenvalues >= threshold * lambda_max))
    return ConditioningResult(
        eigenvalues=eigenvalues,
        condition_number=condition_number,
        effective_rank=rank,
        n_parameters=eigenvalues.size,
    )
