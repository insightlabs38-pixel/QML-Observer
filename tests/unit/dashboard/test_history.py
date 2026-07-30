"""Unit tests for qml_observer.dashboard.history (Milestone 11, Issue #81)."""

from __future__ import annotations

from qml_observer.core.monitor import QMLMonitor
from qml_observer.dashboard.history import discover_run_history
from qml_observer.reporting.reporter import RunReporter


def _finalize_run(path, *, framework="generic", planned_steps=20, n_steps=5):
    reporter = RunReporter(path, framework=framework, planned_steps=planned_steps)
    monitor = QMLMonitor(reporter=reporter, planned_steps=planned_steps)
    for step in range(n_steps):
        monitor.update(step=step, loss=1.0 / (step + 1))
    monitor.finish()
    return reporter


class TestDiscoverRunHistory:
    def test_missing_directory_returns_empty(self, tmp_path):
        assert discover_run_history(tmp_path / "does-not-exist") == []

    def test_empty_directory_returns_empty(self, tmp_path):
        assert discover_run_history(tmp_path) == []

    def test_finds_finalized_runs(self, tmp_path):
        for i in range(3):
            _finalize_run(tmp_path / f"run{i}.jsonl", n_steps=5 + i)

        entries = discover_run_history(tmp_path)
        assert len(entries) == 3
        assert all(e.issue == "insufficient_evidence" for e in entries)
        assert {e.steps for e in entries} == {5, 6, 7}

    def test_skips_unfinished_run(self, tmp_path):
        _finalize_run(tmp_path / "done.jsonl")

        # Started but never finished -- no "summary" record yet.
        unfinished_reporter = RunReporter(tmp_path / "unfinished.jsonl")
        unfinished_monitor = QMLMonitor(reporter=unfinished_reporter)
        unfinished_monitor.update(step=0, loss=1.0)

        entries = discover_run_history(tmp_path)
        assert len(entries) == 1
        assert entries[0].path.endswith("done.jsonl")

    def test_skips_unparseable_file_without_raising(self, tmp_path):
        _finalize_run(tmp_path / "good.jsonl")
        (tmp_path / "garbage.jsonl").write_text("not json at all\n{{{\n")

        entries = discover_run_history(tmp_path)
        assert len(entries) == 1
        assert entries[0].path.endswith("good.jsonl")

    def test_entries_sorted_newest_first(self, tmp_path):
        import os
        import time

        _finalize_run(tmp_path / "older.jsonl")
        time.sleep(0.01)
        _finalize_run(tmp_path / "newer.jsonl")
        # Ensure the two files have distinct mtimes even on coarse filesystems.
        older_stat = os.stat(tmp_path / "older.jsonl")
        newer_stat = os.stat(tmp_path / "newer.jsonl")
        assert newer_stat.st_mtime >= older_stat.st_mtime

        entries = discover_run_history(tmp_path)
        assert entries[0].path.endswith("newer.jsonl")
        assert entries[-1].path.endswith("older.jsonl")

    def test_respects_custom_pattern(self, tmp_path):
        _finalize_run(tmp_path / "run.jsonl")
        _finalize_run(tmp_path / "run.log")  # different extension, ignored by default

        assert len(discover_run_history(tmp_path)) == 1
        assert len(discover_run_history(tmp_path, pattern="*.log")) == 1

    def test_to_dict_roundtrips_expected_keys(self, tmp_path):
        _finalize_run(tmp_path / "run.jsonl", framework="qiskit")
        entry = discover_run_history(tmp_path)[0]
        payload = entry.to_dict()
        assert payload["framework"] == "qiskit"
        assert set(payload.keys()) == {
            "path",
            "run_id",
            "framework",
            "steps",
            "issue",
            "confidence",
            "severity",
            "degraded",
            "estimated_compute_saved",
            "formatted_compute_saved",
            "duration",
            "modified_at",
        }
