"""Reporting: JSONL event logging, run summaries, and compute-saved estimates.

Milestone 7 (Volume XII), Issues #48, #49, #51. Milestone 14, Issue #103b
adds `qml_observer.reporting.history` (run comparison / experiment
management, plan.md §25).
"""

from qml_observer.reporting.export import (
    estimate_compute_saved,
    estimate_compute_saved_from_state,
    export_summary_json,
    format_compute_saved,
)
from qml_observer.reporting.history import (
    HistoryReporter,
    RunHistory,
    RunRecord,
    compare_runs,
    format_comparison_table,
)
from qml_observer.reporting.jsonl import JSONLWriter, read_jsonl
from qml_observer.reporting.reporter import RunReporter
from qml_observer.reporting.summary import build_run_summary

__all__ = [
    "JSONLWriter",
    "read_jsonl",
    "RunReporter",
    "build_run_summary",
    "estimate_compute_saved",
    "estimate_compute_saved_from_state",
    "export_summary_json",
    "format_compute_saved",
    "RunHistory",
    "RunRecord",
    "HistoryReporter",
    "compare_runs",
    "format_comparison_table",
]
