"""JSONL event/diagnosis logging.

Milestone 7 (Volume XII), Issue #48 ("JSONL event logging").

Blueprint scope note: `TrainingEvent` (Milestone 1) is deliberately the
framework-agnostic core event -- gradient/circuit/optimizer metadata live
on `StepObservation` (`core/events.py`) one layer up, and `QMLMonitor`'s
`reporter` hook (Milestone 2, Issue #11) only ever calls
`record_event(event)` with the bare `TrainingEvent` (see
`core/monitor.py::update`), not the full observation. This module's
`event_record`/`diagnosis_record` helpers therefore serialize exactly what
the reporter hook actually receives; callers who want gradient/circuit/
optimizer detail in their JSONL log should log a `StepObservation` (or its
pieces) directly via `JSONLWriter.write()`, using the `*_to_dict` helpers
below.

Every dataclass is converted to a plain JSON-safe `dict` before writing:
enums become their `.value`, and `GradientSnapshot.values` (a numpy
array) is included only when `include_gradient_values=True` is passed
explicitly, since raw arrays can make logs large fast and most consumers
only need the summary statistics.
Milestone 15, Issue #108 (pulled forward -- see
`future_milestones_plan.md`'s "Gaps & recommendations" #7: "worth doing
earlier than 'hardening patch'... retrofitting a version field onto years
of existing 0.x logs later is much more annoying than adding one now,
before those fields exist"): every record now carries a `schema_version`
field (`JSONL_SCHEMA_VERSION`), so a future field addition/rename to any
record type can be distinguished from an older log without guessing from
the field set alone. Bumping `JSONL_SCHEMA_VERSION` is itself a
documented, versioned event -- same rule as addendum §3's threshold
recalibration: a `CHANGELOG.md` entry and a note in this module's
docstring for what changed and why.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.gradient import GradientSnapshot
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent

#: `type` field used for each JSONL record kind, so a reader (`read_jsonl`,
#: the CLI) can distinguish them without guessing from the field set.
RECORD_TYPE_EVENT = "event"
RECORD_TYPE_DIAGNOSIS = "diagnosis"
RECORD_TYPE_SUMMARY = "summary"

#: Version of this module's JSONL record shapes (Issue #108). Bump this
#: whenever a field is added, renamed, or removed from any record type,
#: and add a note here plus a `CHANGELOG.md` entry explaining what changed:
#:
#: - `1` (Milestone 9): initial versioned schema. Introduced alongside the
#:   `ci_lower`/`ci_upper`/`ci_level`/`ci_method` fields on the `gradient`
#:   sub-record (Issue #69) -- there is no unversioned "version 0" log
#:   format in the wild to stay compatible with, since v0.1.0 shipped
#:   without a version field and v0.2.0 introduced the first versioned
#:   schema; `read_jsonl`/the CLI treat a record with no `schema_version`
#:   key as implicitly pre-1 (the v0.1.0 shape) for that reason.
JSONL_SCHEMA_VERSION = 1


def training_event_to_dict(event: TrainingEvent) -> dict[str, Any]:
    """Serialize a `TrainingEvent` to a plain, JSON-safe dict."""
    return {
        "run_id": event.run_id,
        "step": event.step,
        "loss": event.loss,
        "epoch": event.epoch,
        "timestamp": event.timestamp,
        "wall_time": event.wall_time,
    }


def gradient_snapshot_to_dict(
    gradient: GradientSnapshot | None,
    *,
    include_values: bool = False,
) -> dict[str, Any] | None:
    """Serialize a `GradientSnapshot` to a plain, JSON-safe dict.

    `values` (the raw gradient array) is omitted by default -- pass
    `include_values=True` to retain it (as a plain list) for logs where
    full reproducibility matters more than log size.
    """
    if gradient is None:
        return None
    record: dict[str, Any] = {
        "norm_l2": gradient.norm_l2,
        "mean_abs": gradient.mean_abs,
        "variance": gradient.variance,
        "min_value": gradient.min_value,
        "max_value": gradient.max_value,
        "median_abs": gradient.median_abs,
        "snr": gradient.snr,
        "uncertainty": gradient.uncertainty,
        "method": gradient.method,
        "ci_lower": gradient.ci_lower,
        "ci_upper": gradient.ci_upper,
        "ci_level": gradient.ci_level,
        "ci_method": gradient.ci_method,
    }
    if include_values and gradient.values is not None:
        record["values"] = gradient.values.tolist()
    return record


def circuit_metadata_to_dict(circuit: CircuitMetadata | None) -> dict[str, Any] | None:
    """Serialize a `CircuitMetadata` to a plain, JSON-safe dict."""
    if circuit is None:
        return None
    return {
        "n_qubits": circuit.n_qubits,
        "depth": circuit.depth,
        "n_parameters": circuit.n_parameters,
        "n_gates": circuit.n_gates,
        "n_entangling_gates": circuit.n_entangling_gates,
        "ansatz_name": circuit.ansatz_name,
        "initialization": circuit.initialization,
    }


def optimizer_metadata_to_dict(optimizer: OptimizerMetadata | None) -> dict[str, Any] | None:
    """Serialize an `OptimizerMetadata` to a plain, JSON-safe dict."""
    if optimizer is None:
        return None
    return {
        "name": optimizer.name,
        "learning_rate": optimizer.learning_rate,
        "gradient_method": optimizer.gradient_method,
    }


def diagnosis_result_to_dict(diagnosis: DiagnosisResult) -> dict[str, Any]:
    """Serialize a `DiagnosisResult` to a plain, JSON-safe dict."""
    return {
        "issue": diagnosis.issue.value,
        "confidence": diagnosis.confidence,
        "severity": diagnosis.severity,
        "evidence": list(diagnosis.evidence),
        "recommendations": list(diagnosis.recommendations),
        "degraded": diagnosis.degraded,
        "degraded_reason": diagnosis.degraded_reason,
    }


def event_record(
    event: TrainingEvent,
    *,
    gradient: GradientSnapshot | None = None,
    circuit: CircuitMetadata | None = None,
    optimizer: OptimizerMetadata | None = None,
    shots: int | None = None,
    include_gradient_values: bool = False,
) -> dict[str, Any]:
    """Build one `"event"`-type JSONL record.

    `gradient`/`circuit`/`optimizer`/`shots` are optional extras beyond
    what `QMLMonitor`'s automatic reporter hook provides (see module
    docstring) -- pass them when logging a full `StepObservation`
    (e.g. from `state.latest_observation`) rather than a bare `event`.
    """
    record: dict[str, Any] = {
        "type": RECORD_TYPE_EVENT,
        "schema_version": JSONL_SCHEMA_VERSION,
        **training_event_to_dict(event),
    }
    if gradient is not None:
        record["gradient"] = gradient_snapshot_to_dict(
            gradient, include_values=include_gradient_values
        )
    if circuit is not None:
        record["circuit"] = circuit_metadata_to_dict(circuit)
    if optimizer is not None:
        record["optimizer"] = optimizer_metadata_to_dict(optimizer)
    if shots is not None:
        record["shots"] = shots
    return record


def diagnosis_record(diagnosis: DiagnosisResult, *, step: int | None = None) -> dict[str, Any]:
    """Build one `"diagnosis"`-type JSONL record."""
    record: dict[str, Any] = {
        "type": RECORD_TYPE_DIAGNOSIS,
        "schema_version": JSONL_SCHEMA_VERSION,
    }
    if step is not None:
        record["step"] = step
    record.update(diagnosis_result_to_dict(diagnosis))
    return record


def summary_record(summary: dict[str, Any]) -> dict[str, Any]:
    """Wrap a run-summary dict (see `reporting.summary.build_run_summary`
    or `reporting.reporter.RunReporter.finalize`) as a `"summary"`-type
    JSONL record, so a reader can distinguish it from per-step records."""
    return {
        "type": RECORD_TYPE_SUMMARY,
        "schema_version": JSONL_SCHEMA_VERSION,
        **summary,
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class JSONLWriter:
    """Appends dict records as newline-delimited JSON to a file.

    Not thread-safe, consistent with the rest of the monitoring core
    (addendum, Concurrency / Distributed Training) -- use one writer per
    process/rank. Opens the file in append mode and creates parent
    directories as needed, so repeated runs against the same path
    accumulate rather than clobber (callers that want a fresh log per run
    should use a fresh/unique path, e.g. one per `run_id`).

    Example:
        >>> with JSONLWriter("run.jsonl") as writer:
        ...     writer.write(event_record(event))
        ...     writer.write(diagnosis_record(diagnosis, step=event.step))
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        """Serialize `record` as one JSON line and flush immediately.

        Flushing every write trades a little throughput for durability:
        a crashed training loop should still leave a readable partial log
        (matching the project's overall fail-open/transparency stance,
        addendum §1), rather than losing buffered lines.
        """
        self._fh.write(json.dumps(record, default=_json_default))
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JSONLWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL log file back into an iterator of dict records.

    Blank lines are skipped. Used by `reporting.reporter`'s summary
    reconstruction helpers and by the `inspect`/`report` CLI subcommands
    (Issue #50).
    """
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
