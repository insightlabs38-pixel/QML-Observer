"""Parameter-redundancy detection in the ansatz.

Milestone 12 (blueprint Volume XIII), Issue #85, built on `conditioning.py`
(Issue #84).

## Mathematical description

A near-zero QFIM eigenvalue `lambda_k` with eigenvector `v_k` identifies a
direction in parameter space, `sum_i v_k[i] * e_i`, that produces
(to first order) no change in the output state at all. When `v_k` is
concentrated on a small number of parameters (a few large components,
the rest near zero), that is direct evidence those specific parameters
are locally redundant -- e.g. two rotation gates that commute and can be
merged, or a rotation whose axis has been made irrelevant by the
surrounding circuit structure at this particular point in parameter
space. This is a genuinely different, complementary signal to a flat
loss/gradient: it explains a *structural* reason a subset of parameters
might be hard to train, rather than just reporting that training overall
looks flat.

## Method

For every eigenvector associated with a near-zero eigenvalue (per
`conditioning.effective_rank`'s threshold), find the parameter indices
whose squared component in that eigenvector exceeds a separate
`contribution_threshold` (default `0.1`, i.e. contributing at least 10%
of that direction's squared norm -- eigenvectors are unit-normalized by
`numpy.linalg.eigvalsh`/`eigh`, so squared components already sum to 1
and are directly comparable across eigenvectors of different circuits).
Report the union of those indices as "redundant parameter" candidates,
along with which null-space direction(s) implicate each one, so a user
can inspect the specific gates involved.

This is deliberately a *candidate* list, not a certainty: a parameter
appearing here means "locally redundant at the sampled point," not
"always redundant" -- see Known Limitations in `docs/research/geometry.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qml_observer.advanced.geometry.conditioning import _eigenvalues_descending


@dataclass
class RedundancyResult:
    """Candidate locally-redundant parameters found in a QFIM's null space.

    Attributes:
        redundant_parameter_indices: Sorted, deduplicated indices (into
            the parameter vector the QFIM was computed at) flagged as
            contributing meaningfully to at least one near-zero-eigenvalue
            direction.
        null_space_dimension: Number of near-zero eigenvalues found
            (`n_parameters - effective_rank`).
        effective_rank: The QFIM's effective rank (see `conditioning.
            effective_rank`), included here so a caller doesn't need a
            second call to relate `null_space_dimension` back to it.
        contributions: Maps each flagged parameter index to the list of
            null-space eigenvector indices (0 = smallest eigenvalue, in
            ascending-eigenvalue order) it contributes to, for a user who
            wants to inspect the specific null-space direction(s)
            involved rather than just the flat parameter list.
    """

    redundant_parameter_indices: list[int]
    null_space_dimension: int
    effective_rank: int
    contributions: dict[int, list[int]] = field(default_factory=dict)


def detect_redundant_parameters(
    qfim: np.ndarray,
    rank_threshold: float = 1e-6,
    contribution_threshold: float = 0.1,
) -> RedundancyResult:
    """Detect parameters that locally contribute to the QFIM's null space.

    Args:
        qfim: A square, symmetric QFIM (e.g. from
            `qfim.estimate_qfim`).
        rank_threshold: Relative eigenvalue cutoff defining "near-zero",
            passed through to the same role as `conditioning.
            effective_rank`'s `threshold` -- kept as a separate parameter
            name here (`rank_threshold`) rather than reusing
            `threshold` to avoid ambiguity with `contribution_threshold`
            below, which is a different, unrelated cutoff.
        contribution_threshold: Minimum squared eigenvector component
            (fraction of that null-space direction's unit norm) for a
            parameter to be flagged as contributing to it. Must be in
            `(0, 1]`.

    Returns:
        A `RedundancyResult`. `redundant_parameter_indices` is empty
        (not an error) when the QFIM is full-rank -- "no locally
        redundant parameters found" is itself a meaningful, common
        result, not a degenerate case.

    Raises:
        ValueError: If `qfim` is not square/non-empty (via the same
            check `conditioning` functions use), or if
            `contribution_threshold` is not in `(0, 1]`.
    """
    if not (0.0 < contribution_threshold <= 1.0):
        raise ValueError(f"contribution_threshold must be in (0, 1], got {contribution_threshold}")

    eigenvalues_desc = _eigenvalues_descending(qfim)
    lambda_max = eigenvalues_desc[0]
    n = eigenvalues_desc.size

    if lambda_max <= 0:
        # Fully degenerate QFIM: every parameter is trivially "redundant"
        # (nothing changes the state at all). Report the whole index set
        # rather than silently returning empty, since that would read as
        # "no redundancy found," the opposite of the truth here.
        return RedundancyResult(
            redundant_parameter_indices=list(range(n)),
            null_space_dimension=n,
            effective_rank=0,
            contributions={i: [] for i in range(n)},
        )

    # eigh returns eigenvalues ascending with matching eigenvectors as
    # columns -- recomputed here (rather than reusing eigvalsh's result)
    # because eigenvectors are needed, which eigvalsh does not provide.
    eigenvalues_asc, eigenvectors = np.linalg.eigh(qfim)
    eigenvalues_asc = np.clip(eigenvalues_asc, 0.0, None)

    null_space_idx = [k for k in range(n) if eigenvalues_asc[k] < rank_threshold * lambda_max]

    contributions: dict[int, list[int]] = {}
    for null_k in null_space_idx:
        vec = eigenvectors[:, null_k]
        squared = np.abs(vec) ** 2
        for param_idx in np.where(squared >= contribution_threshold)[0]:
            contributions.setdefault(int(param_idx), []).append(null_k)

    redundant = sorted(contributions.keys())
    return RedundancyResult(
        redundant_parameter_indices=redundant,
        null_space_dimension=len(null_space_idx),
        effective_rank=n - len(null_space_idx),
        contributions=contributions,
    )
