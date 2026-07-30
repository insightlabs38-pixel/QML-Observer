"""Dashboard data sources.

Milestone 11, Issue #76 ("Dashboard architecture decision + scaffold").

The dashboard is a *read* layer on top of data that already exists
elsewhere in the project (`docs/roadmap.md`'s own framing: "serving JSON
from existing `RunReporter`/JSONL data"). It does not introduce a new
place metrics are computed -- it introduces a `DashboardDataSource`
interface so the FastAPI app (`dashboard/app.py`) can read from whichever
source the caller already has, without caring which one it is:

- `MonitorDataSource` wraps a live `QMLMonitor`. This is the richest
  source: `monitor.state.window` holds the bounded window of full
  `StepObservation`s (loss *and* gradient snapshots), so both the loss
  chart (Issue #77) and the gradient chart (Issue #78) can be served from
  a single in-process object while training is actually running.
- `ReporterDataSource` wraps a `RunReporter`. Per that module's own
  docstring, `QMLMonitor`'s automatic reporter hook only ever forwards
  the bare `TrainingEvent` (loss, no gradient) -- so this source can serve
  the loss chart and the diagnosis/compute-usage panels, but its gradient
  series is always empty. Documented, not silently guessed at.
- `JSONLDataSource` wraps a JSONL log file (`reporting/jsonl.py`) for
  post-hoc/offline viewing of a completed (or in-progress, since the
  writer flushes every line) run. Gradient data is only present in the
  series if the log's `event` records happen to include a `"gradient"`
  sub-record -- i.e. if the caller logged full `StepObservation` detail
  themselves (see that module's docstring), not just the automatic hook.

All three expose the same four read methods, so `dashboard/app.py`'s
routes never need to know which kind of source they were given.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qml_observer.reporting.export import estimate_compute_saved, format_compute_saved
from qml_observer.reporting.jsonl import (
    RECORD_TYPE_DIAGNOSIS,
    RECORD_TYPE_EVENT,
    RECORD_TYPE_SUMMARY,
    read_jsonl,
)
from qml_observer.schemas.diagnosis import DiagnosisResult

if TYPE_CHECKING:
    from qml_observer.core.monitor import QMLMonitor
    from qml_observer.reporting.reporter import RunReporter


@dataclass
class LossSeries:
    """Loss-over-step data for the live loss chart (Issue #77)."""

    steps: list[int] = field(default_factory=list)
    loss: list[float | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "loss": self.loss}


@dataclass
class GradientSeries:
    """Gradient-statistics-over-step data for the gradient chart (Issue #78).

    Empty (all-empty lists) when the underlying source has no per-step
    gradient detail available -- see module docstring. Callers should
    treat an empty series as "no gradient data available", not "gradient
    is zero".
    """

    steps: list[int] = field(default_factory=list)
    norm_l2: list[float | None] = field(default_factory=list)
    variance: list[float | None] = field(default_factory=list)
    snr: list[float | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "norm_l2": self.norm_l2,
            "variance": self.variance,
            "snr": self.snr,
        }

    @property
    def available(self) -> bool:
        return len(self.steps) > 0


def _diagnosis_to_dict(diagnosis: DiagnosisResult | None) -> dict[str, Any] | None:
    """Shared serialization for the diagnosis panel (Issue #79).

    Mirrors `reporting.jsonl.diagnosis_result_to_dict` but lives here
    (rather than importing it) since a dict-shaped diagnosis pulled from a
    `summary` JSONL record uses the same key set already and doesn't need
    round-tripping through a `DiagnosisResult` instance.
    """
    if diagnosis is None:
        return None
    return {
        "issue": diagnosis.issue.value,
        "confidence": diagnosis.confidence,
        "severity": diagnosis.severity,
        "evidence": list(diagnosis.evidence),
        "recommendations": list(diagnosis.recommendations),
        "degraded": diagnosis.degraded,
        "degraded_reason": diagnosis.degraded_reason,
    }


@dataclass
class ComputeUsage:
    """Compute-usage panel data (Issue #80).

    `estimated_compute_saved`/`formatted` follow the addendum §11 formula
    exactly (`reporting.export.estimate_compute_saved`): `None` (never a
    fabricated guess) when `planned_steps` or a mean per-step wall time is
    unavailable.
    """

    run_id: str | None
    framework: str | None
    actual_steps: int
    planned_steps: int | None
    mean_wall_time_per_step: float | None
    estimated_compute_saved: float | None
    formatted: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "framework": self.framework,
            "actual_steps": self.actual_steps,
            "planned_steps": self.planned_steps,
            "mean_wall_time_per_step": self.mean_wall_time_per_step,
            "estimated_compute_saved": self.estimated_compute_saved,
            "formatted": self.formatted,
        }


class DashboardDataSource(ABC):
    """Read-only interface the dashboard app reads from.

    Implementations must be safe to poll repeatedly (the frontend polls
    these on an interval, see `dashboard/static/app.js`) and must never
    raise on ordinary "no data yet" states -- an empty/default result is
    always preferred to an exception, consistent with the project's
    fail-open posture (addendum §1) extended to this read path.
    """

    @abstractmethod
    def loss_series(self) -> LossSeries: ...

    @abstractmethod
    def gradient_series(self) -> GradientSeries: ...

    @abstractmethod
    def diagnosis(self) -> dict[str, Any] | None: ...

    @abstractmethod
    def compute_usage(self) -> ComputeUsage: ...

    @abstractmethod
    def run_id(self) -> str | None: ...


class MonitorDataSource(DashboardDataSource):
    """Reads live from an attached `QMLMonitor` (the richest source).

    Example:
        >>> monitor = QMLMonitor(planned_steps=1000)
        >>> source = MonitorDataSource(monitor, framework="pennylane")
        >>> app = create_app(source)
    """

    def __init__(self, monitor: QMLMonitor, *, framework: str | None = None) -> None:
        self._monitor = monitor
        self._framework = framework

    def loss_series(self) -> LossSeries:
        window = self._monitor.state.window
        return LossSeries(
            steps=[obs.training_event.step for obs in window],
            loss=[obs.training_event.loss for obs in window],
        )

    def gradient_series(self) -> GradientSeries:
        window = [obs for obs in self._monitor.state.window if obs.gradient is not None]
        return GradientSeries(
            steps=[obs.training_event.step for obs in window],
            norm_l2=[obs.gradient.norm_l2 for obs in window],  # type: ignore[union-attr]
            variance=[obs.gradient.variance for obs in window],  # type: ignore[union-attr]
            snr=[obs.gradient.snr for obs in window],  # type: ignore[union-attr]
        )

    def diagnosis(self) -> dict[str, Any] | None:
        return _diagnosis_to_dict(self._monitor.latest_diagnosis())

    def compute_usage(self) -> ComputeUsage:
        state = self._monitor.state
        mean_wall_time = state.mean_wall_time()
        saved = estimate_compute_saved(state.planned_steps, state.step_count, mean_wall_time)
        return ComputeUsage(
            run_id=state.run_id,
            framework=self._framework,
            actual_steps=state.step_count,
            planned_steps=state.planned_steps,
            mean_wall_time_per_step=mean_wall_time,
            estimated_compute_saved=saved,
            formatted=format_compute_saved(saved),
        )

    def run_id(self) -> str | None:
        return self._monitor.run_id


class ReporterDataSource(DashboardDataSource):
    """Reads from a `RunReporter` (loss + diagnosis + compute usage only).

    `gradient_series()` always returns an empty `GradientSeries` -- a
    `RunReporter` fed purely through `QMLMonitor`'s automatic hook never
    receives gradient detail (see `reporting/reporter.py`'s docstring).
    Use `MonitorDataSource` instead if a live gradient chart matters.
    """

    def __init__(self, reporter: RunReporter, *, framework: str | None = None) -> None:
        self._reporter = reporter
        self._framework = framework

    def loss_series(self) -> LossSeries:
        events = self._reporter.events
        return LossSeries(
            steps=[e.step for e in events],
            loss=[e.loss for e in events],
        )

    def gradient_series(self) -> GradientSeries:
        return GradientSeries()

    def diagnosis(self) -> dict[str, Any] | None:
        diagnoses = self._reporter.diagnoses
        return _diagnosis_to_dict(diagnoses[-1]) if diagnoses else None

    def compute_usage(self) -> ComputeUsage:
        summary = self._reporter.summary
        events = self._reporter.events
        run_id = events[-1].run_id if events else None
        wall_times = [e.wall_time for e in events if e.wall_time is not None]
        mean_wall_time = (sum(wall_times) / len(wall_times)) if wall_times else None
        planned_steps = getattr(self._reporter, "_planned_steps", None)
        saved = summary["estimated_compute_saved"] if summary is not None else None
        return ComputeUsage(
            run_id=run_id,
            framework=self._framework,
            actual_steps=len(events),
            planned_steps=planned_steps,
            mean_wall_time_per_step=mean_wall_time,
            estimated_compute_saved=saved,
            formatted=format_compute_saved(saved),
        )

    def run_id(self) -> str | None:
        events = self._reporter.events
        return events[-1].run_id if events else None


class JSONLDataSource(DashboardDataSource):
    """Reads from a JSONL run log file, re-reading it on every call.

    Re-reading on each call (rather than caching) trades a little I/O for
    always reflecting the latest line the training process has flushed --
    `JSONLWriter` flushes every write (`reporting/jsonl.py`), so this is
    effectively "live" for a run that's still writing to `path`, and exact
    for a completed one.
    """

    def __init__(self, path: str | Path, *, framework: str | None = None) -> None:
        self._path = Path(path)
        self._framework = framework

    def _records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        return list(read_jsonl(self._path))

    def loss_series(self) -> LossSeries:
        events = [r for r in self._records() if r.get("type") == RECORD_TYPE_EVENT]
        return LossSeries(
            steps=[int(r["step"]) for r in events],
            loss=[r.get("loss") for r in events],
        )

    def gradient_series(self) -> GradientSeries:
        events = [
            r for r in self._records() if r.get("type") == RECORD_TYPE_EVENT and r.get("gradient")
        ]
        return GradientSeries(
            steps=[int(r["step"]) for r in events],
            norm_l2=[r["gradient"].get("norm_l2") for r in events],
            variance=[r["gradient"].get("variance") for r in events],
            snr=[r["gradient"].get("snr") for r in events],
        )

    def diagnosis(self) -> dict[str, Any] | None:
        records = self._records()
        summaries = [r for r in records if r.get("type") == RECORD_TYPE_SUMMARY]
        diagnoses = [r for r in records if r.get("type") == RECORD_TYPE_DIAGNOSIS]
        final = summaries[-1] if summaries else (diagnoses[-1] if diagnoses else None)
        if final is None:
            return None
        return {
            "issue": final.get("issue") or final.get("final_diagnosis"),
            "confidence": final.get("confidence"),
            "severity": final.get("severity"),
            "evidence": final.get("evidence") or [],
            "recommendations": final.get("recommendations") or [],
            "degraded": bool(final.get("degraded", False)),
            "degraded_reason": final.get("degraded_reason"),
        }

    def compute_usage(self) -> ComputeUsage:
        """Compute-usage panel data reconstructed from the JSONL log.

        Note: `reporting.jsonl.summary_record`/`RunReporter._build_summary`
        do not currently carry a `planned_steps` field of their own (only
        the already-computed `estimated_compute_saved`) -- so
        `ComputeUsage.planned_steps` is always `None` when read from a
        JSONL log, even though `estimated_compute_saved` itself correctly
        reflects whatever `planned_steps` was configured at record time.
        Use `MonitorDataSource`/`ReporterDataSource` directly (in-process)
        if displaying the configured `planned_steps` value matters, or
        pass one through `JSONLDataSource(path, ...)` explicitly in a
        future revision if this gap needs closing.
        """
        records = self._records()
        events = [r for r in records if r.get("type") == RECORD_TYPE_EVENT]
        summaries = [r for r in records if r.get("type") == RECORD_TYPE_SUMMARY]
        summary = summaries[-1] if summaries else None
        run_id = events[-1].get("run_id") if events else None
        wall_times: list[float] = [
            float(w) for r in events if (w := r.get("wall_time")) is not None and math.isfinite(w)
        ]
        mean_wall_time = (sum(wall_times) / len(wall_times)) if wall_times else None
        planned_steps = summary.get("planned_steps") if summary else None
        saved = summary.get("estimated_compute_saved") if summary else None
        return ComputeUsage(
            run_id=run_id,
            framework=(summary.get("framework") if summary else None) or self._framework,
            actual_steps=len(events),
            planned_steps=planned_steps,
            mean_wall_time_per_step=mean_wall_time,
            estimated_compute_saved=saved,
            formatted=format_compute_saved(saved),
        )

    def run_id(self) -> str | None:
        events = [r for r in self._records() if r.get("type") == RECORD_TYPE_EVENT]
        return events[-1].get("run_id") if events else None
