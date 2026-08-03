"""RunHistory: a local, append-only ledger of completed-run summaries.

Milestone 14 (`future_milestones_plan.md`), Issue #103b ("Run comparison /
experiment management", plan.md §25). plan.md scopes run comparison, A/B
testing of ansatzes, baseline comparison, saved run history, and
CSV/JSON export as their own feature area; this module implements all of
them at the level actually specified (a local ledger plus simple
comparison/export), without inventing an experiment-tracking UI plan.md
never asked for -- that's what Issue #101's `MLflowTracker`/`WandbTracker`
are for if you want one.

`RunHistory` is deliberately independent of, and complementary to,
per-step JSONL logging (`RunReporter`/`reporting/jsonl.py`, Milestone 7):
it stores exactly one line per *completed* run (the same summary shape
`RunReporter.finalize()`/`reporting/summary.build_run_summary()` already
produce), tagged with arbitrary user metadata (e.g. `{"ansatz":
"hardware_efficient"}`), so many runs can be listed, filtered, and
compared -- the "saved run history" and "A/B testing of ansatzes" cases
from plan.md §25.

Typical usage, tagging runs so they can be compared later:

    from qml_observer.reporting.history import RunHistory, HistoryReporter

    history = RunHistory("experiments/history.jsonl")
    reporter = HistoryReporter(history, tags={"ansatz": "hardware_efficient"})
    monitor = QMLMonitor(reporter=reporter)
    ...
    monitor.finish()

    # Later, comparing every run tagged with a given ansatz:
    runs = history.filter_by_tag("ansatz", "hardware_efficient")
    for row in compare_runs(runs):
        print(row)
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qml_observer.reporting.reporter import RunReporter

#: Version of `RunRecord`'s on-disk JSON shape, following the same
#: documented-and-versioned convention as `reporting/jsonl.py`'s
#: `JSONL_SCHEMA_VERSION` (Issue #108) -- bump and note here if
#: `RunRecord`'s fields ever change.
HISTORY_SCHEMA_VERSION = 1

#: Columns written by `RunHistory.export_csv()`, and the row order
#: `compare_runs()`/`format_comparison_table()` use. `tags` is
#: intentionally last and JSON-encoded (a CSV cell can't hold a nested
#: dict) since its keys vary run to run.
_COMPARISON_COLUMNS = (
    "run_id",
    "framework",
    "final_diagnosis",
    "confidence",
    "severity",
    "steps",
    "duration",
    "estimated_compute_saved",
    "degraded",
    "recorded_at",
    "tags",
)


@dataclass
class RunRecord:
    """One completed run's summary, as stored in a `RunHistory` ledger.

    Mirrors the fields `RunReporter.finalize()`
    (`reporting/reporter.py::_build_summary`) already produces, plus
    `tags` (arbitrary user-supplied metadata for later filtering/
    comparison -- e.g. ansatz name, framework, experiment notes) and
    `recorded_at` (when this record was appended, not when the run
    itself started/ended, since that's already covered by `duration`).
    """

    run_id: str | None
    framework: str | None
    steps: int
    duration: float | None
    final_diagnosis: str | None
    confidence: float | None
    severity: str | None
    degraded: bool
    estimated_compute_saved: float | None
    tags: dict[str, str] = field(default_factory=dict)
    recorded_at: float = field(default_factory=time.time)

    @classmethod
    def from_summary(cls, summary: dict[str, Any], tags: dict[str, str] | None = None) -> RunRecord:
        """Build a `RunRecord` from a `RunReporter.finalize()`-shaped summary dict.

        Only the fields `RunRecord` actually stores are read; extra keys
        in `summary` (e.g. `evidence`, `recommendations`,
        `loss_curve_summary`, `degraded_reason`) are intentionally not
        carried into the history ledger -- those live in the per-run
        JSONL log already, and `RunHistory` is meant to stay a compact,
        many-rows-at-a-glance comparison ledger rather than duplicating
        the full per-run record.
        """
        return cls(
            run_id=summary.get("run_id"),
            framework=summary.get("framework"),
            steps=summary.get("steps", 0),
            duration=summary.get("duration"),
            final_diagnosis=summary.get("final_diagnosis"),
            confidence=summary.get("confidence"),
            severity=summary.get("severity"),
            degraded=bool(summary.get("degraded", False)),
            estimated_compute_saved=summary.get("estimated_compute_saved"),
            tags=dict(tags) if tags else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-safe dict, including the schema version."""
        return {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "framework": self.framework,
            "steps": self.steps,
            "duration": self.duration,
            "final_diagnosis": self.final_diagnosis,
            "confidence": self.confidence,
            "severity": self.severity,
            "degraded": self.degraded,
            "estimated_compute_saved": self.estimated_compute_saved,
            "tags": dict(self.tags),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Parse a `RunRecord` back from `to_dict()`'s JSON shape.

        Ignores an unrecognized/future `schema_version` rather than
        raising -- a forward-compatible reader is more useful than a
        strict one for a local ledger a person may keep appending to
        across `qml-observer` versions.
        """
        return cls(
            run_id=data.get("run_id"),
            framework=data.get("framework"),
            steps=data.get("steps", 0),
            duration=data.get("duration"),
            final_diagnosis=data.get("final_diagnosis"),
            confidence=data.get("confidence"),
            severity=data.get("severity"),
            degraded=bool(data.get("degraded", False)),
            estimated_compute_saved=data.get("estimated_compute_saved"),
            tags=dict(data.get("tags") or {}),
            recorded_at=data.get("recorded_at", 0.0),
        )


class RunHistory:
    """An append-only, local JSONL ledger of `RunRecord`s.

    Distinct from a per-step JSONL run log (`RunReporter`'s `jsonl_path`):
    this file accumulates one line per *completed* run, across however
    many separate training runs you've done, so it can answer "how do my
    last 10 runs compare" rather than "what happened during this one
    run".
    """

    def __init__(self, path: str | Path) -> None:
        """Open (or prepare to create) a history ledger at `path`.

        Args:
            path: File path for the JSONL ledger. Parent directories are
                created on first `append()` if they don't already exist;
                nothing is written or read at construction time.
        """
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: RunRecord) -> None:
        """Append one `RunRecord` as a new line. Never overwrites existing lines."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def append_summary(
        self, summary: dict[str, Any], tags: dict[str, str] | None = None
    ) -> RunRecord:
        """Convenience: build a `RunRecord.from_summary(...)` and `append()` it."""
        record = RunRecord.from_summary(summary, tags)
        self.append(record)
        return record

    def load_all(self) -> list[RunRecord]:
        """Read every record currently in the ledger, oldest first.

        Returns an empty list (rather than raising) if the file doesn't
        exist yet -- a ledger with nothing appended to it yet is a normal
        starting state, not an error.
        """
        if not self._path.exists():
            return []
        records = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(RunRecord.from_dict(json.loads(line)))
        return records

    def get(self, run_id: str) -> RunRecord | None:
        """Return the most recently appended record for `run_id`, or `None`.

        The *most recent* match is returned (not the first) in case a
        `run_id` was ever reused or re-recorded, so this reflects the
        latest known state for that run.
        """
        match: RunRecord | None = None
        for record in self.load_all():
            if record.run_id == run_id:
                match = record
        return match

    def filter_by_tag(self, key: str, value: str) -> list[RunRecord]:
        """Return every record whose `tags[key] == value`, oldest first."""
        return [r for r in self.load_all() if r.tags.get(key) == value]

    def export_csv(self, path: str | Path) -> None:
        """Write every record to `path` as CSV, one row per run.

        `tags` is JSON-encoded into a single cell (see
        `_COMPARISON_COLUMNS`'s docstring note) since CSV has no native
        nested-value representation.
        """
        records = self.load_all()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_COMPARISON_COLUMNS)
            writer.writeheader()
            for record in records:
                writer.writerow(_comparison_row(record))

    def export_json(self, path: str | Path) -> None:
        """Write every record to `path` as a single JSON array."""
        records = self.load_all()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, indent=2, sort_keys=True)


def _comparison_row(record: RunRecord) -> dict[str, Any]:
    row = record.to_dict()
    row["tags"] = json.dumps(record.tags, sort_keys=True) if record.tags else ""
    return {col: row.get(col) for col in _COMPARISON_COLUMNS}


def compare_runs(records: Sequence[RunRecord]) -> list[dict[str, Any]]:
    """Build a flat comparison table (list of row dicts) from a set of records.

    Each row has exactly `_COMPARISON_COLUMNS`'s keys, with `tags`
    rendered as a JSON string (matching `export_csv()`'s cell format) so
    the same row shape works for both a CSV export and a plain-text
    table (`format_comparison_table`). Rows preserve `records`' input
    order -- callers wanting a specific order (e.g. by recency, by
    confidence) should sort `records` themselves first.
    """
    return [_comparison_row(record) for record in records]


def format_comparison_table(records: Sequence[RunRecord]) -> str:
    """Render `compare_runs(records)` as a simple, fixed-width text table.

    Intended for terminal/CLI output (`qml-observer history compare`);
    for anything else (a notebook, a dashboard), read `compare_runs()`'s
    row dicts directly instead of parsing this string.
    """
    if not records:
        return "No runs to compare."

    columns = (
        "run_id",
        "final_diagnosis",
        "confidence",
        "steps",
        "estimated_compute_saved",
        "tags",
    )
    rows = [_comparison_row(r) for r in records]

    def _cell(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    widths = {
        col: max(len(col), *(len(_cell(row[col])) for row in rows)) for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    separator = "  ".join("-" * widths[col] for col in columns)
    lines = [header, separator]
    for row in rows:
        lines.append("  ".join(_cell(row[col]).ljust(widths[col]) for col in columns))
    return "\n".join(lines)


class HistoryReporter:
    """A `RunReporter`-duck-typed wrapper that also appends the finished run
    to a `RunHistory` ledger.

    Wraps an internal `RunReporter` (built with no `jsonl_path`, so it
    only accumulates in-memory events/diagnoses to build the summary --
    pass your own separately-configured `RunReporter` as `QMLMonitor`'s
    reporter alongside this one, or use
    `qml_observer.integrations.trackers`' fan-out pattern, if you also
    want per-step JSONL logging from the same run) rather than
    duplicating its summary-building logic. Appending to `history` is
    wrapped in the same fail-open policy as `BaseExperimentTracker`
    (addendum §1): a full/unwritable ledger file must never propagate
    into the training loop.

    Example:
        >>> history = RunHistory("experiments/history.jsonl")
        >>> monitor = QMLMonitor(
        ...     reporter=HistoryReporter(history, tags={"ansatz": "hardware_efficient"})
        ... )
    """

    def __init__(
        self,
        history: RunHistory,
        *,
        tags: dict[str, str] | None = None,
        framework: str | None = None,
        planned_steps: int | None = None,
    ) -> None:
        """Create a reporter appending to `history` at `finalize()` time.

        Args:
            history: The `RunHistory` ledger to append this run's summary
                to once it finishes.
            tags: Arbitrary metadata for this run (e.g. `{"ansatz": ...}`),
                stored on the resulting `RunRecord` for later filtering/
                comparison.
            framework: Passed straight through to the internal
                `RunReporter` (see `RunReporter.__init__`).
            planned_steps: Passed straight through to the internal
                `RunReporter`, for its compute-saved estimate.
        """
        self._history = history
        self._tags = dict(tags) if tags else {}
        self._inner = RunReporter(framework=framework, planned_steps=planned_steps)
        self._appended = False

    def record_event(self, event: Any) -> None:
        self._inner.record_event(event)

    def record_diagnosis(self, diagnosis: Any) -> None:
        self._inner.record_diagnosis(diagnosis)

    def finalize(self) -> dict[str, Any]:
        summary = self._inner.finalize()
        if self._appended:
            return summary
        try:
            self._history.append_summary(summary, tags=self._tags)
        except Exception:
            import logging

            logging.getLogger("qml_observer.reporting.history").warning(
                "qml_observer: failed to append run to history ledger at %s; "
                "training already completed uninterrupted.",
                self._history.path,
                exc_info=True,
            )
        self._appended = True
        return summary
