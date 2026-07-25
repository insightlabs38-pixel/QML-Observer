"""Unit tests for qml_observer.reporting.reporter.RunReporter (Milestone 7, Issue #49)."""

from qml_observer.core.monitor import QMLMonitor
from qml_observer.reporting.jsonl import read_jsonl
from qml_observer.reporting.reporter import RunReporter


class TestRunReporterDuckType:
    def test_wires_into_monitor_end_to_end(self):
        reporter = RunReporter(framework="generic", planned_steps=10)
        monitor = QMLMonitor(reporter=reporter, planned_steps=10)

        for step in range(5):
            monitor.update(step=step, loss=1.0 - step * 0.1)
        monitor.finish()

        assert len(reporter.events) == 5
        assert len(reporter.diagnoses) == 1
        assert reporter.summary is not None
        assert reporter.summary["run_id"] == monitor.run_id
        assert reporter.summary["steps"] == 5
        assert reporter.summary["framework"] == "generic"

    def test_finalize_is_idempotent(self):
        reporter = RunReporter()
        monitor = QMLMonitor(reporter=reporter)
        monitor.update(step=0, loss=1.0)
        monitor.finish()
        first = reporter.finalize()
        second = reporter.finalize()
        assert first == second

    def test_summary_none_before_finalize(self):
        reporter = RunReporter()
        assert reporter.summary is None


class TestRunReporterJSONLStreaming:
    def test_streams_events_diagnoses_and_summary_to_file(self, tmp_path):
        path = tmp_path / "run.jsonl"
        reporter = RunReporter(path, planned_steps=10)
        monitor = QMLMonitor(reporter=reporter, planned_steps=10)

        for step in range(3):
            monitor.update(step=step, loss=1.0)
        monitor.finish()

        records = list(read_jsonl(path))
        types = [r["type"] for r in records]
        assert types.count("event") == 3
        assert types.count("diagnosis") == 1
        assert types.count("summary") == 1

    def test_summary_record_has_compute_saved(self, tmp_path):
        path = tmp_path / "run.jsonl"
        reporter = RunReporter(path, planned_steps=100)
        monitor = QMLMonitor(reporter=reporter, planned_steps=100)
        for step in range(5):
            monitor.update(step=step, loss=1.0)
        monitor.finish()

        records = list(read_jsonl(path))
        summary_record = next(r for r in records if r["type"] == "summary")
        assert "estimated_compute_saved" in summary_record


class TestRunReporterWithoutData:
    def test_finalize_with_no_events_or_diagnoses(self):
        reporter = RunReporter()
        summary = reporter.finalize()
        assert summary["run_id"] is None
        assert summary["steps"] == 0
        assert summary["final_diagnosis"] is None
        assert summary["degraded"] is False
        assert summary["evidence"] == []


class TestRunReporterFailureIsolation:
    def test_broken_writer_path_does_not_crash_construction(self, tmp_path):
        # Sanity: nested, not-yet-existing directories are created rather
        # than raising (mirrors JSONLWriter's own guarantee).
        path = tmp_path / "a" / "b" / "c" / "run.jsonl"
        reporter = RunReporter(path)
        reporter.finalize()
        assert path.exists()
