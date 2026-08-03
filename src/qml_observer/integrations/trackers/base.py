"""Shared plumbing for experiment-tracker integrations (Milestone 14, Issue #101).

`MLflowTracker` and `WandbTracker` (in their own sibling modules, each
gated behind its own optional dependency) both implement the `RunReporter`
duck type `QMLMonitor(reporter=...)` already knows how to drive
(`record_event`, `record_diagnosis`, `finalize` -- see
`reporting/reporter.py`'s module docstring for the exact contract), and
both translate the same qml_observer schema objects
(`TrainingEvent`/`DiagnosisResult`) into that tracker's own metric-logging
calls. `BaseExperimentTracker` here factors out everything that isn't
tracker-specific: which fields to log, and the fail-open error handling
around actually logging them.

Neither this module nor its subclasses replace `RunReporter`/JSONL logging
(Milestone 7) -- they're an additional sink, not a competing one. If you
want both, pass a tracker in place of `RunReporter` for tracker-only
projects, or write a small fan-out reporter:

    class FanOutReporter:
        def __init__(self, *reporters):
            self._reporters = reporters
        def record_event(self, event):
            for r in self._reporters:
                r.record_event(event)
        def record_diagnosis(self, diagnosis):
            for r in self._reporters:
                r.record_diagnosis(diagnosis)
        def finalize(self):
            return [r.finalize() for r in self._reporters][0]

    reporter = FanOutReporter(RunReporter("run.jsonl"), MLflowTracker())
    monitor = QMLMonitor(reporter=reporter)
"""

from __future__ import annotations

import logging
from typing import Any

from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.training import TrainingEvent

_logger = logging.getLogger("qml_observer.integrations.trackers")


def event_metrics(event: TrainingEvent) -> dict[str, float]:
    """Extract the numeric, step-level metrics worth logging from a `TrainingEvent`.

    Fields that are `None` are omitted entirely rather than logged as
    `0`/`NaN` -- most trackers (MLflow, W&B) treat a step with no logged
    value for a metric differently from one explicitly logged as `0`.
    """
    metrics: dict[str, float] = {}
    if event.loss is not None:
        metrics["loss"] = float(event.loss)
    if event.wall_time is not None:
        metrics["wall_time"] = float(event.wall_time)
    return metrics


def diagnosis_metrics(diagnosis: DiagnosisResult) -> dict[str, Any]:
    """Extract the fields worth logging from a final `DiagnosisResult`."""
    return {
        "final_issue": diagnosis.issue.value,
        "confidence": diagnosis.confidence,
        "severity": diagnosis.severity,
        "degraded": diagnosis.degraded,
    }


class BaseExperimentTracker:
    """Common `record_event`/`record_diagnosis`/`finalize` skeleton for trackers.

    Subclasses implement `_log_metrics(step, metrics)` (called with
    per-step numeric metrics, and once more at `record_diagnosis()` time
    with `{"confidence": ...}`) and `_log_summary(summary)` (called once
    at `finalize()` with the dict from `diagnosis_metrics()`).

    Both call sites are wrapped in a fail-open try/except here, mirroring
    `QMLMonitor`'s own addendum §1 policy: a tracker being unreachable
    (network issue, disabled/finished run, missing credentials, etc.) must
    never propagate into the caller's training loop, which by this point
    has already survived `QMLMonitor.update()`'s own fail-open handling
    once. Failures are logged at `warning` level, matching the rest of the
    project's fail-open logging.
    """

    def __init__(self) -> None:
        self._step_count = 0
        self._last_diagnosis: DiagnosisResult | None = None

    # -- RunReporter duck type (called by QMLMonitor) ----------------------

    def record_event(self, event: TrainingEvent) -> None:
        """Log this step's numeric metrics. Called once per `QMLMonitor.update()`."""
        self._step_count = event.step
        metrics = event_metrics(event)
        if not metrics:
            return
        try:
            self._log_metrics(event.step, metrics)
        except Exception:
            _logger.warning(
                "qml_observer: experiment tracker failed to log metrics for "
                "step=%s; training continues uninterrupted.",
                event.step,
                exc_info=True,
            )

    def record_diagnosis(self, diagnosis: DiagnosisResult) -> None:
        """Log the final `DiagnosisResult`'s confidence. Called once by `finish()`."""
        self._last_diagnosis = diagnosis
        try:
            self._log_metrics(self._step_count, {"confidence": diagnosis.confidence})
        except Exception:
            _logger.warning(
                "qml_observer: experiment tracker failed to log final diagnosis; "
                "training continues uninterrupted.",
                exc_info=True,
            )

    def finalize(self) -> dict[str, Any]:
        """Log the run summary and return it; idempotent-safe to call once by `finish()`."""
        summary = diagnosis_metrics(self._last_diagnosis) if self._last_diagnosis else {}
        try:
            self._log_summary(summary)
        except Exception:
            _logger.warning(
                "qml_observer: experiment tracker failed to finalize run summary.",
                exc_info=True,
            )
        return summary

    # -- subclass hooks ------------------------------------------------------

    def _log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        raise NotImplementedError

    def _log_summary(self, summary: dict[str, Any]) -> None:
        raise NotImplementedError
