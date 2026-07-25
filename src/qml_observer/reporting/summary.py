"""Run summary reports.

Milestone 7 (Volume XII / plan.md §14), Issue #49.

`build_run_summary()` is the full-fidelity summary builder: given a
`RunState` (e.g. `monitor.state`) and a final `DiagnosisResult` (e.g.
`monitor.finish()`), it produces the blueprint's Volume XII output shape

    {
        "run_id": ...,
        "duration": ...,
        "steps": ...,
        "final_diagnosis": ...,
        "confidence": ...,
        "estimated_compute_saved": ...,
    }

extended with plan.md §14's additional fields (circuit metadata,
optimizer, shot budget, gradient statistics, loss curve summary,
evidence/recommendations) pulled from `state.latest_observation` and
`state.window`.

Scope note: `QMLMonitor`'s automatic `reporter` hook (Milestone 2,
`core/monitor.py`) only ever calls `record_event(event)` with the bare,
framework-agnostic `TrainingEvent` -- not the full `StepObservation` --
so `reporting.reporter.RunReporter` (fed purely through that hook) cannot
by itself recover gradient/circuit/optimizer detail. `build_run_summary`
is the richer, direct-call alternative for exactly that reason: call it
with `monitor.state` once training finishes, rather than relying solely
on the automatic reporter wiring.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from qml_observer.reporting.export import estimate_compute_saved_from_state
from qml_observer.reporting.jsonl import (
    circuit_metadata_to_dict,
    gradient_snapshot_to_dict,
    optimizer_metadata_to_dict,
)
from qml_observer.schemas.diagnosis import DiagnosisResult

if TYPE_CHECKING:
    from qml_observer.core.state import RunState


def _loss_curve_summary(losses: list[float]) -> dict[str, Any] | None:
    if not losses:
        return None
    finite = [v for v in losses if math.isfinite(v)]
    return {
        "n_points": len(losses),
        "first": losses[0],
        "last": losses[-1],
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "has_non_finite": len(finite) != len(losses),
    }


def build_run_summary(
    state: RunState,
    diagnosis: DiagnosisResult,
    *,
    framework: str | None = None,
) -> dict[str, Any]:
    """Build a full run-summary dict from `state` and a final `diagnosis`.

    Args:
        state: The run's `RunState` (e.g. `monitor.state`), read for
            timing, the rolling observation window, and `planned_steps`.
        diagnosis: The diagnosis to report as `final_diagnosis` (e.g. the
            return value of `monitor.finish()`).
        framework: Optional label for the framework/adapter used (e.g.
            `"pennylane"`, `"qiskit"`, `"generic"`), for the report header.

    Returns:
        A JSON-safe dict combining the blueprint's Volume XII fields with
        plan.md §14's circuit/optimizer/gradient/loss-curve detail. Fields
        with no available data are `None` rather than omitted, so
        consumers (CLI/report/dashboard) can rely on a stable key set.
    """
    duration = None
    if state.start_time is not None and state.end_time is not None:
        duration = state.end_time - state.start_time

    latest = state.latest_observation
    circuit = latest.circuit if latest is not None else None
    optimizer = latest.optimizer if latest is not None else None
    shots = latest.shots if latest is not None else None
    gradient = latest.gradient if latest is not None else None

    losses = [
        obs.training_event.loss for obs in state.window if obs.training_event.loss is not None
    ]

    return {
        "run_id": state.run_id,
        "framework": framework,
        "duration": duration,
        "steps": state.step_count,
        "final_diagnosis": diagnosis.issue.value,
        "confidence": diagnosis.confidence,
        "severity": diagnosis.severity,
        "degraded": diagnosis.degraded,
        "degraded_reason": diagnosis.degraded_reason,
        "circuit": circuit_metadata_to_dict(circuit),
        "optimizer": optimizer_metadata_to_dict(optimizer),
        "shots": shots,
        "gradient": gradient_snapshot_to_dict(gradient),
        "loss_curve_summary": _loss_curve_summary(losses),
        "evidence": list(diagnosis.evidence),
        "recommendations": list(diagnosis.recommendations),
        "estimated_compute_saved": estimate_compute_saved_from_state(state),
    }
