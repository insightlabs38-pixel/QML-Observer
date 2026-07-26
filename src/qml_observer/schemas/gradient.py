"""GradientSnapshot schema.

A GradientSnapshot is a structured, framework-agnostic summary of the
gradient observed at a single training step. Adapters (or user code, via
the generic adapter) are expected to call `summarize_gradient` once per
step rather than have every detector recompute the same reductions over
the raw gradient array.

Note on scope: this module intentionally does *not* attempt to classify
NaN/Inf gradients as an "unstable" signal, and does not implement rolling
persistence logic — that classification is a detector-layer concern
(Milestone 4) built on top of the statistics engine (Milestone 3). This
module's job is limited to producing a correct, honestly-labeled summary
of whatever array it was given.
"""

import math
from dataclasses import dataclass

import numpy as np

from qml_observer.schemas._validation import check_non_negative_number, check_type


@dataclass
class GradientSnapshot:
    """Summary statistics for a gradient observed at one training step.

    Attributes:
        values: The raw gradient array as provided, or None if the caller
            chose not to retain raw values (e.g. to keep telemetry/logs
            small). Summary statistics below are always computed from the
            array at snapshot-creation time, independent of whether it is
            retained here.
        norm_l2: L2 (Euclidean) norm of the flattened gradient.
        mean_abs: Mean of the absolute gradient values.
        variance: Variance of the (signed) gradient values.
        min_value: Minimum gradient value.
        max_value: Maximum gradient value.
        median_abs: Median of the absolute gradient values.
        snr: Estimated signal-to-noise ratio, if available (populated by
            the statistics engine, Milestone 3 / Volume IV, not by
            `summarize_gradient` itself).
        uncertainty: Estimated measurement uncertainty, if available (also
            populated by the statistics engine).
        method: Name of the gradient computation method (e.g.
            "parameter-shift", "adjoint", "finite-difference"), if known.
    """

    values: np.ndarray | None
    norm_l2: float
    mean_abs: float
    variance: float
    min_value: float
    max_value: float
    median_abs: float
    snr: float | None = None
    uncertainty: float | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        if self.values is not None:
            check_type(self.values, np.ndarray, "values")
        # norm/variance/spread stats are magnitudes: negative is never
        # valid, but NaN/Inf are tolerated (diverging/degenerate gradients
        # are meaningful signal, not a schema error — addendum §7).
        check_non_negative_number(self.norm_l2, "norm_l2")
        check_non_negative_number(self.mean_abs, "mean_abs")
        check_non_negative_number(self.variance, "variance")
        check_non_negative_number(self.median_abs, "median_abs")
        check_type(self.min_value, (int, float), "min_value")
        check_type(self.max_value, (int, float), "max_value")
        if not (math.isnan(self.min_value) or math.isnan(self.max_value)):
            if self.min_value > self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) must be <= max_value ({self.max_value})"
                )
        check_non_negative_number(self.snr, "snr")
        check_non_negative_number(self.uncertainty, "uncertainty")
        if self.method is not None:
            check_type(self.method, str, "method")


def summarize_gradient(
    gradients: np.ndarray,
    method: str | None = None,
    *,
    keep_values: bool = True,
) -> GradientSnapshot:
    """Compute a GradientSnapshot from a raw gradient array.

    Args:
        gradients: Array-like of gradient values (any shape; flattened
            internally for reduction statistics).
        method: Name of the gradient computation method, if known.
        keep_values: If False, the returned snapshot's `values` field is
            None (useful for keeping logs/telemetry small); the raw array
            is still used to compute statistics either way.

    Returns:
        A populated GradientSnapshot. `snr` and `uncertainty` are left
        unset here; they are filled in by the statistics engine, which
        additionally needs shot-count / measurement-variance context that
        this function does not have.

    Raises:
        ValueError: If `gradients` is empty. An empty gradient array
            usually indicates an adapter misconfiguration upstream, and
            per the project's fail-open policy this must surface as a
            clear, catchable error here rather than an opaque numpy
            exception (e.g. "zero-size array to reduction operation").
    """
    array = np.asarray(gradients, dtype=float)

    if array.size == 0:
        raise ValueError(
            "summarize_gradient() received an empty gradient array. "
            "This usually indicates a misconfigured adapter or a training "
            "loop that failed to compute gradients for this step."
        )

    flat = array.ravel()
    abs_flat = np.abs(flat)

    # Inf/NaN entries are an explicitly anticipated case (addendum §7: a
    # diverging optimizer's gradient may legitimately contain them, and
    # that is itself the signal `DiagnosisEngine._check_instability` acts
    # on) -- not a bug to warn about. Suppress numpy's informational
    # RuntimeWarning for the resulting inf-inf=nan/inf*0=nan arithmetic
    # here, matching `statistics.loss.relative_loss_improvement`'s same
    # treatment of the same expected edge case.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        norm_l2 = float(np.linalg.norm(flat, ord=2))
        mean_abs = float(np.mean(abs_flat))
        variance = float(np.var(flat))
        median_abs = float(np.median(abs_flat))

    return GradientSnapshot(
        values=array if keep_values else None,
        norm_l2=norm_l2,
        mean_abs=mean_abs,
        variance=variance,
        min_value=float(np.min(flat)),
        max_value=float(np.max(flat)),
        median_abs=median_abs,
        method=method,
    )
