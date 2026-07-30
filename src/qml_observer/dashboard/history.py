"""Run history across multiple JSONL logs (Milestone 11, Issue #81).

Every other dashboard data source (`data_source.py`) shows exactly one
run. This module adds the complementary "many runs" view: given a
directory of JSONL logs (one per run, as written by `RunReporter`/
`JSONLWriter`), build a lightweight summary table without needing to load
every run's full event history into the dashboard at once.

Only *finalized* runs are included -- a log with no `"summary"` record
yet (i.e. `RunReporter.finalize()`/`QMLMonitor.finish()` hasn't been
called for it) is a run still in progress, which the live
`MonitorDataSource`/single-run view already covers; it is intentionally
left out of the history table rather than shown with placeholder blanks.

Reading is fail-open, matching addendum §1 extended to this read path: a
single malformed or half-written log file is skipped (with a warning
printed to stderr) rather than breaking the history view for every other
run in the directory.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qml_observer.reporting.export import format_compute_saved
from qml_observer.reporting.jsonl import RECORD_TYPE_SUMMARY, read_jsonl


@dataclass
class RunHistoryEntry:
    """One row of the run-history table, built from a run's final
    `"summary"` JSONL record (see `reporting/reporter.py::_build_summary`
    for exactly which fields that record carries)."""

    path: str
    run_id: str | None
    framework: str | None
    steps: int
    issue: str | None
    confidence: float | None
    severity: str | None
    degraded: bool
    estimated_compute_saved: float | None
    formatted_compute_saved: str
    duration: float | None
    modified_at: str
    """ISO 8601 UTC timestamp of the log file's last modification time --
    a proxy for "when this run happened", since the summary record itself
    does not carry a wall-clock start time. Good enough for sorting/
    display; not a substitute for `TrainingEvent.timestamp` if exact
    provenance matters."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "run_id": self.run_id,
            "framework": self.framework,
            "steps": self.steps,
            "issue": self.issue,
            "confidence": self.confidence,
            "severity": self.severity,
            "degraded": self.degraded,
            "estimated_compute_saved": self.estimated_compute_saved,
            "formatted_compute_saved": self.formatted_compute_saved,
            "duration": self.duration,
            "modified_at": self.modified_at,
        }


def _entry_from_file(file_path: Path) -> RunHistoryEntry | None:
    records = list(read_jsonl(file_path))
    summaries = [r for r in records if r.get("type") == RECORD_TYPE_SUMMARY]
    if not summaries:
        return None
    summary = summaries[-1]
    saved = summary.get("estimated_compute_saved")
    mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)
    return RunHistoryEntry(
        path=str(file_path),
        run_id=summary.get("run_id"),
        framework=summary.get("framework"),
        steps=summary.get("steps", 0),
        issue=summary.get("final_diagnosis"),
        confidence=summary.get("confidence"),
        severity=summary.get("severity"),
        degraded=bool(summary.get("degraded", False)),
        estimated_compute_saved=saved,
        formatted_compute_saved=format_compute_saved(saved),
        duration=summary.get("duration"),
        modified_at=mtime.isoformat(),
    )


def discover_run_history(
    directory: str | Path,
    *,
    pattern: str = "*.jsonl",
) -> list[RunHistoryEntry]:
    """Scan `directory` for finalized-run JSONL logs and summarize each.

    Args:
        directory: Directory to scan (non-recursively) for logs.
        pattern: Glob pattern for log files. Defaults to `"*.jsonl"`.

    Returns:
        `RunHistoryEntry` list, newest (`modified_at`) first. Empty if
        `directory` doesn't exist, is empty, or contains no finalized
        runs -- never raises for those ordinary "no history yet" cases.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    entries: list[RunHistoryEntry] = []
    for file_path in sorted(directory.glob(pattern)):
        if not file_path.is_file():
            continue
        try:
            entry = _entry_from_file(file_path)
        except Exception as exc:  # noqa: BLE001 - fail-open, see module docstring
            print(
                f"warning: qml-observer dashboard could not read run history "
                f"from {file_path}: {exc}",
                file=sys.stderr,
            )
            continue
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda e: e.modified_at, reverse=True)
    return entries
