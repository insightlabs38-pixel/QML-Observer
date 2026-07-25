"""RunReporter: the blueprint's Volume XII reporter duck type.

Milestone 7, Issue #49 ("Run summary reports"), building on Issue #48's
JSONL logging (`reporting/jsonl.py`) and Issue #51's compute-saved
estimate (`reporting/export.py`).

`QMLMonitor(reporter=...)` (Milestone 2) calls exactly three methods on
whatever object it's given: `record_event(event)` once per `update()`
call, `record_diagnosis(diagnosis)` once at `finish()`, and `finalize()`
once at the end of `finish()` -- all best-effort (fail-open: a raising
reporter never propagates into the training loop, addendum §1).
`RunReporter` implements that exact contract.

Because `record_event` only ever receives the bare `TrainingEvent` (see
`core/monitor.py::update` and `core/events.py`'s module docstring for why
gradient/circuit/optimizer live one layer up on `StepObservation`
instead), the summary `RunReporter.finalize()` can build from its own
recorded history is necessarily limited to run identity, per-step
timing/loss, and the final diagnosis. For the fuller plan.md §14 report
(circuit/optimizer metadata, gradient statistics), call
`reporting.summary.build_run_summary(monitor.state, final_diagnosis)`
directly instead, using `monitor.state` -- see that module's docstring.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from qml_observer.reporting.export import estimate_compute_saved
from qml_observer.reporting.jsonl import (
    JSONLWriter,
    diagnosis_record,
    event_record,
    summary_record,
)
from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.training import TrainingEvent


class RunReporter:
    """Accumulates events/diagnoses for a run and produces a summary.

    Implements the blueprint's `record_event`/`record_diagnosis`/
    `finalize` duck type (Volume XII) so it can be passed directly as
    `QMLMonitor(reporter=RunReporter(...))`.

    Example:
        >>> reporter = RunReporter("runs/run.jsonl", framework="pennylane",
        ...                        planned_steps=1000)
        >>> monitor = QMLMonitor(reporter=reporter, planned_steps=1000)
        >>> for step in range(1000):
        ...     diagnosis = monitor.update(step=step, loss=loss)
        ...     if monitor.should_stop():
        ...         break
        >>> monitor.finish()
        >>> reporter.summary["estimated_compute_saved"]
    """

    def __init__(
        self,
        jsonl_path: str | Path | None = None,
        *,
        framework: str | None = None,
        planned_steps: int | None = None,
    ) -> None:
        """Create a reporter, optionally streaming to a JSONL log.

        Args:
            jsonl_path: If given, every recorded event/diagnosis (and the
                final summary) is additionally appended to this file as
                JSON lines (Issue #48), readable via
                `reporting.jsonl.read_jsonl` or the `qml-observer inspect`/
                `report` CLI subcommands (Issue #50).
            framework: Optional label for the framework/adapter in use
                (e.g. `"pennylane"`, `"qiskit"`), included in the summary.
            planned_steps: Total steps this run is expected to take, used
                for the compute-saved estimate (Issue #51). Should
                normally match whatever was passed to
                `QMLMonitor(planned_steps=...)`.
        """
        self._events: list[TrainingEvent] = []
        self._diagnoses: list[DiagnosisResult] = []
        self._framework = framework
        self._planned_steps = planned_steps
        self._writer = JSONLWriter(jsonl_path) if jsonl_path is not None else None
        self._finalized = False
        self._summary: dict[str, Any] | None = None

    # -- RunReporter duck type (called by QMLMonitor) ----------------------

    def record_event(self, event: TrainingEvent) -> None:
        """Record one `TrainingEvent`. Called once per `QMLMonitor.update()`."""
        self._events.append(event)
        if self._writer is not None:
            self._writer.write(event_record(event))

    def record_diagnosis(self, diagnosis: DiagnosisResult) -> None:
        """Record the final `DiagnosisResult`. Called once by `finish()`."""
        self._diagnoses.append(diagnosis)
        if self._writer is not None:
            step = self._events[-1].step if self._events else None
            self._writer.write(diagnosis_record(diagnosis, step=step))

    def finalize(self) -> dict[str, Any]:
        """Build and return the run summary; idempotent.

        Also appends a `"summary"`-type JSONL record (if a `jsonl_path`
        was configured) and closes the writer, since `finalize()` is only
        ever called once by `QMLMonitor.finish()`.
        """
        if self._finalized:
            assert self._summary is not None
            return self._summary

        self._summary = self._build_summary()
        if self._writer is not None:
            self._writer.write(summary_record(self._summary))
            self._writer.close()
        self._finalized = True
        return self._summary

    # -- read access ---------------------------------------------------------

    @property
    def summary(self) -> dict[str, Any] | None:
        """The summary dict from `finalize()`, or `None` if not finalized yet."""
        return self._summary

    @property
    def events(self) -> list[TrainingEvent]:
        """All `TrainingEvent`s recorded so far, oldest first."""
        return list(self._events)

    @property
    def diagnoses(self) -> list[DiagnosisResult]:
        """All `DiagnosisResult`s recorded so far, oldest first."""
        return list(self._diagnoses)

    # -- summary construction -------------------------------------------------

    def _mean_wall_time(self) -> float | None:
        wall_times = [e.wall_time for e in self._events if e.wall_time is not None]
        if not wall_times:
            return None
        return sum(wall_times) / len(wall_times)

    def _loss_curve_summary(self) -> dict[str, Any] | None:
        losses = [e.loss for e in self._events if e.loss is not None]
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

    def _build_summary(self) -> dict[str, Any]:
        run_id = self._events[-1].run_id if self._events else None
        duration = (
            (self._events[-1].timestamp - self._events[0].timestamp)
            if len(self._events) >= 2
            and self._events[0].timestamp is not None
            and self._events[-1].timestamp is not None
            else None
        )
        final = self._diagnoses[-1] if self._diagnoses else None

        return {
            "run_id": run_id,
            "framework": self._framework,
            "duration": duration,
            "steps": len(self._events),
            "final_diagnosis": final.issue.value if final is not None else None,
            "confidence": final.confidence if final is not None else None,
            "severity": final.severity if final is not None else None,
            "degraded": final.degraded if final is not None else False,
            "degraded_reason": final.degraded_reason if final is not None else None,
            "loss_curve_summary": self._loss_curve_summary(),
            "evidence": list(final.evidence) if final is not None else [],
            "recommendations": list(final.recommendations) if final is not None else [],
            "estimated_compute_saved": estimate_compute_saved(
                self._planned_steps, len(self._events), self._mean_wall_time()
            ),
        }
