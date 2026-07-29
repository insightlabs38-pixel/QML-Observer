"""Synthetic training-run generators for detector/diagnosis validation.

Milestone 4 (Volume VI / Volume XVIII), Issue #32.

These generators produce deterministic (seeded), framework-agnostic
sequences of per-step keyword arguments -- each dict is suitable for
`QMLMonitor.update(**step)` directly -- covering the benchmark categories
from plan.md §15:

1. `healthy_learning_run`    -- gradients stay informative, loss decreases
   steadily, but training has not yet reached a "good enough" absolute
   loss to call it converged.
2. `convergence_run`         -- loss decays toward a low absolute value
   while the gradient shrinks in step, consistent with settling into a
   good optimum (as opposed to collapsing early).
3. `artificial_plateau_run`  -- loss stuck at a poor value from the start,
   gradients collapsed from the start: the barren-plateau shape.
4. `noise_dominated_run`     -- large gradient variance and a
   fluctuating-but-not-trending loss; used to check detectors do *not*
   false-positive on noise alone (addendum §3's false-positive concern,
   ahead of the dedicated `NoiseDetector` in Milestone 9).
5. `stagnant_optimizer_run`  -- learning rate pinned at 0.0 with frozen
   parameters, independent of gradient magnitude.
6. `diverging_optimizer_run` -- loss and gradients run finite for a while,
   then go NaN/Inf, as a real optimizer divergence would. Used to confirm
   `DiagnosisEngine` reports `IssueType.UNSTABLE` rather than silently
   letting NaN propagate into a false `HEALTHY` reading (addendum §7,
   closed during Milestone 7 beta review -- see
   `diagnosis/engine.py::_check_instability`).

Not included here: the plan.md §15 "depth scaling case", which requires
comparing gradient-variance trends *across* multiple circuit
configurations (qubit count / depth) rather than within a single run --
that is `ScalingAnalyzer` territory (blueprint Volume XIII, Milestone 12)
and needs the circuit-scaling infrastructure built there.

These fixtures are deliberately framework-agnostic (no PennyLane/Qiskit
dependency) so they can be reused by unit tests (`tests/unit/`), the
benchmark suite (`benchmarks/`, Milestone 7), and the empirical
calibration process (addendum §3) without any adapter in the loop.

Every generator accepts `seed` for reproducibility and `n_steps`/`n_params`
to control run length and parameter-vector size.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qml_observer.schemas.optimizer import OptimizerMetadata


def healthy_learning_run(
    n_steps: int = 60, n_params: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """Steadily decreasing loss with informative (non-collapsed) gradients.

    Loss decays geometrically by a random factor each step; gradient scale
    tracks the current loss (larger loss -> larger, more informative
    gradient), consistent with healthy but not-yet-converged learning.
    """
    rng = np.random.default_rng(seed)
    loss = 2.0
    steps = []
    for i in range(n_steps):
        loss = max(0.05, loss * rng.uniform(0.93, 0.99))
        gradient = rng.normal(0, max(0.05, loss * 0.3), size=n_params)
        steps.append(
            dict(
                step=i,
                loss=float(loss),
                gradients=gradient,
                optimizer=OptimizerMetadata(name="adam", learning_rate=0.1),
            )
        )
    return steps


def convergence_run(n_steps: int = 80, n_params: int = 10, seed: int = 0) -> list[dict[str, Any]]:
    """Loss decays to a low absolute value; gradient shrinks alongside it.

    Distinguishes "good convergence" from `artificial_plateau_run`: here
    the *loss itself* reaches a low value, not just a flat/stagnant one.
    """
    rng = np.random.default_rng(seed)
    loss = 2.0
    steps = []
    for i in range(n_steps):
        loss = max(0.0005, loss * 0.9)
        gradient_scale = max(2e-5, loss * 0.05)
        gradient = rng.normal(0, gradient_scale, size=n_params)
        steps.append(
            dict(
                step=i,
                loss=float(loss),
                gradients=gradient,
                optimizer=OptimizerMetadata(name="adam", learning_rate=0.1),
            )
        )
    return steps


def artificial_plateau_run(
    n_steps: int = 60, n_params: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """Loss stuck at a poor value from step 0; gradients collapsed from step 0.

    Simulates a bad initialization landing directly on a barren plateau:
    the failure mode is present for the entire run, not developed over time.
    """
    rng = np.random.default_rng(seed)
    base_loss = rng.uniform(0.6, 0.9)
    steps = []
    for i in range(n_steps):
        loss = base_loss + rng.normal(0, 1e-8)
        gradient = rng.normal(0, 1e-6, size=n_params)
        steps.append(
            dict(
                step=i,
                loss=float(loss),
                gradients=gradient,
                optimizer=OptimizerMetadata(name="adam", learning_rate=0.1),
            )
        )
    return steps


def noise_dominated_run(
    n_steps: int = 60, n_params: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """Large, high-variance gradients and a loss that fluctuates without trending.

    No collapse (gradients stay large) and no consistent improvement or
    stagnation -- just noise. Used to confirm detectors do not
    false-positive when the signal is simply noisy rather than genuinely
    failing.
    """
    rng = np.random.default_rng(seed)
    loss = 0.5
    steps = []
    for i in range(n_steps):
        loss = float(np.clip(loss + rng.normal(0, 0.05), 0.05, 1.0))
        gradient = rng.normal(0, 0.5, size=n_params)
        steps.append(
            dict(
                step=i,
                loss=loss,
                gradients=gradient,
                optimizer=OptimizerMetadata(name="adam", learning_rate=0.1),
            )
        )
    return steps


def stagnant_optimizer_run(
    n_steps: int = 50, n_params: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """Learning rate pinned at 0.0 with frozen parameters, gradients notwithstanding.

    Gradients are present and non-trivial in magnitude -- the point is
    that the optimizer itself has stopped applying them, which
    `StagnationDetector` should catch independently of gradient magnitude
    (as distinct from `artificial_plateau_run`).
    """
    rng = np.random.default_rng(seed)
    loss = float(rng.uniform(0.3, 0.7))
    parameters = rng.normal(size=n_params)
    steps = []
    for i in range(n_steps):
        gradient = rng.normal(0, 0.3, size=n_params)
        steps.append(
            dict(
                step=i,
                loss=loss,
                gradients=gradient,
                optimizer=OptimizerMetadata(name="adam", learning_rate=0.0),
                parameters=parameters.copy(),
            )
        )
    return steps


def diverging_optimizer_run(
    n_steps: int = 30, n_params: int = 10, n_finite_steps: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """Finite loss/gradients for a while, then NaN -- a real optimizer divergence.

    Modeled on an exploding-gradient failure: loss grows step over step
    (rather than the healthy/plateau fixtures' shrinking or flat loss)
    before overflowing to NaN, and the gradient norm grows in step with
    it. Used to confirm `DiagnosisEngine` reports `IssueType.UNSTABLE`
    once non-finite values appear, rather than the NaN silently
    propagating into a false `HEALTHY`/`CONVERGED` reading (addendum §7).
    """
    rng = np.random.default_rng(seed)
    loss = 0.5
    steps = []
    for i in range(n_steps):
        if i < n_finite_steps:
            loss = loss * 1.5 + abs(rng.normal(0, 0.05))
            gradient = rng.normal(0, 0.5 * (i + 1), size=n_params)
        else:
            loss = float("nan")
            gradient = np.full(n_params, float("nan"))
        steps.append(
            dict(
                step=i,
                loss=loss,
                gradients=gradient,
                optimizer=OptimizerMetadata(name="adam", learning_rate=0.1),
            )
        )
    return steps


def finite_shots_healthy_run(
    n_steps: int = 60,
    n_params: int = 10,
    shots: int = 20,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """`healthy_learning_run`, but every step also reports a finite `shots`.

    Milestone 9, Issue #68: the false-positive fixture for the shot-noise-
    aware `NoiseDetector` (Milestone 9, Issue #66) -- unlike
    `noise_dominated_run` (which varies gradient *variance* to test the
    existing MVP detectors), this generator holds the underlying learning
    dynamics identical to `healthy_learning_run` and varies only `shots`,
    so it isolates exactly what changes when a real, informative gradient
    is estimated from a small shot budget rather than analytically.
    `shots` is exposed as a parameter specifically so benchmarks can sweep
    it (plan.md §15's "varying shot budgets").
    """
    rng = np.random.default_rng(seed)
    loss = 2.0
    steps = []
    for i in range(n_steps):
        loss = max(0.05, loss * rng.uniform(0.93, 0.99))
        gradient = rng.normal(0, max(0.05, loss * 0.3), size=n_params)
        steps.append(
            dict(
                step=i,
                loss=float(loss),
                gradients=gradient,
                optimizer=OptimizerMetadata(name="adam", learning_rate=0.1),
                shots=shots,
            )
        )
    return steps


def finite_shots_plateau_run(
    n_steps: int = 60,
    n_params: int = 10,
    shots: int = 20,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """`artificial_plateau_run`, but every step also reports a finite `shots`.

    Milestone 9, Issue #68: the true-positive fixture confirming a genuine
    barren-plateau-scale collapse is still recognizable (by
    `BarrenPlateauDetector`) and *not* misclassified as merely
    shot-noise-dominated (by `NoiseDetector`) once a finite shot budget is
    in play -- the gradient's mean magnitude and its variance collapse
    together here, so the shot-noise floor collapses proportionally too
    (see `detectors/noise.py` module docstring for why that keeps the two
    detectors from conflating this case).
    """
    rng = np.random.default_rng(seed)
    base_loss = rng.uniform(0.6, 0.9)
    steps = []
    for i in range(n_steps):
        loss = base_loss + rng.normal(0, 1e-8)
        gradient = rng.normal(0, 1e-6, size=n_params)
        steps.append(
            dict(
                step=i,
                loss=float(loss),
                gradients=gradient,
                optimizer=OptimizerMetadata(name="adam", learning_rate=0.1),
                shots=shots,
            )
        )
    return steps


#: All generators, keyed by scenario name, for parametrized test/benchmark loops.
ALL_SCENARIOS = {
    "healthy_learning": healthy_learning_run,
    "convergence": convergence_run,
    "artificial_plateau": artificial_plateau_run,
    "noise_dominated": noise_dominated_run,
    "stagnant_optimizer": stagnant_optimizer_run,
    "diverging_optimizer": diverging_optimizer_run,
    "finite_shots_healthy": finite_shots_healthy_run,
    "finite_shots_plateau": finite_shots_plateau_run,
}


def run_through_monitor(monitor: Any, steps: list[dict[str, Any]]) -> Any:
    """Feed a generated step list through a `QMLMonitor` and return the final diagnosis.

    Convenience shared by tests and (later) `benchmarks/run_benchmarks.py`
    so both drive the monitor identically.

    Args:
        monitor: A `QMLMonitor` instance (or duck-typed equivalent) with an
            `update(**kwargs)` method.
        steps: A list of step dicts as returned by any generator above.

    Returns:
        The `DiagnosisResult` from the final step. `steps` must be
        non-empty.
    """
    if not steps:
        raise ValueError("steps must be non-empty")
    diagnosis = None
    for step in steps:
        diagnosis = monitor.update(**step)
    return diagnosis
