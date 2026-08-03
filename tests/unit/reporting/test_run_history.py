"""Unit tests for qml_observer.reporting.history (Milestone 14, Issue #103b).

"Run comparison / experiment management" (plan.md §25).
"""

from __future__ import annotations

import json

from qml_observer.core.monitor import QMLMonitor
from qml_observer.reporting.history import (
    HISTORY_SCHEMA_VERSION,
    HistoryReporter,
    RunHistory,
    RunRecord,
    compare_runs,
    format_comparison_table,
)


def _summary(run_id="run-1", framework="pennylane", steps=10, confidence=0.9, degraded=False):
    return {
        "run_id": run_id,
        "framework": framework,
        "duration": 12.5,
        "steps": steps,
        "final_diagnosis": "converged",
        "confidence": confidence,
        "severity": "low",
        "degraded": degraded,
        "degraded_reason": None,
        "loss_curve_summary": {"first": 1.0, "last": 0.01},
        "evidence": ["some evidence"],
        "recommendations": [],
        "estimated_compute_saved": 42.0,
    }


class TestRunRecord:
    def test_from_summary_extracts_expected_fields(self):
        record = RunRecord.from_summary(_summary(), tags={"ansatz": "hea"})
        assert record.run_id == "run-1"
        assert record.framework == "pennylane"
        assert record.steps == 10
        assert record.final_diagnosis == "converged"
        assert record.confidence == 0.9
        assert record.degraded is False
        assert record.estimated_compute_saved == 42.0
        assert record.tags == {"ansatz": "hea"}

    def test_from_summary_defaults_tags_to_empty(self):
        record = RunRecord.from_summary(_summary())
        assert record.tags == {}

    def test_round_trips_through_dict(self):
        record = RunRecord.from_summary(_summary(), tags={"ansatz": "hea"})
        restored = RunRecord.from_dict(record.to_dict())
        assert restored == record

    def test_to_dict_includes_schema_version(self):
        record = RunRecord.from_summary(_summary())
        assert record.to_dict()["schema_version"] == HISTORY_SCHEMA_VERSION

    def test_from_dict_ignores_unknown_schema_version(self):
        data = RunRecord.from_summary(_summary()).to_dict()
        data["schema_version"] = 999
        restored = RunRecord.from_dict(data)
        assert restored.run_id == "run-1"


class TestRunHistoryAppendAndLoad:
    def test_load_all_on_missing_file_returns_empty(self, tmp_path):
        history = RunHistory(tmp_path / "does_not_exist.jsonl")
        assert history.load_all() == []

    def test_append_and_load_all(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        history.append(RunRecord.from_summary(_summary(run_id="a")))
        history.append(RunRecord.from_summary(_summary(run_id="b")))
        records = history.load_all()
        assert [r.run_id for r in records] == ["a", "b"]

    def test_append_summary_convenience(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        record = history.append_summary(_summary(run_id="a"), tags={"ansatz": "hea"})
        assert record.tags == {"ansatz": "hea"}
        assert history.load_all() == [record]

    def test_creates_parent_directories(self, tmp_path):
        history = RunHistory(tmp_path / "nested" / "dir" / "history.jsonl")
        history.append(RunRecord.from_summary(_summary()))
        assert (tmp_path / "nested" / "dir" / "history.jsonl").exists()

    def test_append_is_append_only_across_instances(self, tmp_path):
        path = tmp_path / "history.jsonl"
        RunHistory(path).append(RunRecord.from_summary(_summary(run_id="a")))
        RunHistory(path).append(RunRecord.from_summary(_summary(run_id="b")))
        assert [r.run_id for r in RunHistory(path).load_all()] == ["a", "b"]


class TestRunHistoryGetAndFilter:
    def test_get_returns_matching_record(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        history.append(RunRecord.from_summary(_summary(run_id="a")))
        history.append(RunRecord.from_summary(_summary(run_id="b")))
        record = history.get("b")
        assert record is not None
        assert record.run_id == "b"

    def test_get_returns_none_when_missing(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        history.append(RunRecord.from_summary(_summary(run_id="a")))
        assert history.get("nonexistent") is None

    def test_get_returns_most_recent_on_reused_run_id(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        history.append(RunRecord.from_summary(_summary(run_id="a", confidence=0.1)))
        history.append(RunRecord.from_summary(_summary(run_id="a", confidence=0.9)))
        assert history.get("a").confidence == 0.9

    def test_filter_by_tag(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        history.append_summary(_summary(run_id="a"), tags={"ansatz": "hea"})
        history.append_summary(_summary(run_id="b"), tags={"ansatz": "strong_ent"})
        history.append_summary(_summary(run_id="c"), tags={"ansatz": "hea"})
        matches = history.filter_by_tag("ansatz", "hea")
        assert [r.run_id for r in matches] == ["a", "c"]

    def test_filter_by_tag_no_matches(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        history.append_summary(_summary(run_id="a"), tags={"ansatz": "hea"})
        assert history.filter_by_tag("ansatz", "nonexistent") == []


class TestExport:
    def test_export_json_round_trips(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        history.append_summary(_summary(run_id="a"), tags={"ansatz": "hea"})
        out_path = tmp_path / "export.json"
        history.export_json(out_path)
        data = json.loads(out_path.read_text())
        assert len(data) == 1
        assert data[0]["run_id"] == "a"
        assert data[0]["tags"] == {"ansatz": "hea"}

    def test_export_csv_has_header_and_rows(self, tmp_path):
        import csv

        history = RunHistory(tmp_path / "history.jsonl")
        history.append_summary(_summary(run_id="a"), tags={"ansatz": "hea"})
        history.append_summary(_summary(run_id="b"))
        out_path = tmp_path / "export.csv"
        history.export_csv(out_path)

        with out_path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["run_id"] == "a"
        assert json.loads(rows[0]["tags"]) == {"ansatz": "hea"}
        assert rows[1]["tags"] == ""

    def test_export_creates_parent_directories(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        history.append_summary(_summary())
        history.export_json(tmp_path / "nested" / "out.json")
        assert (tmp_path / "nested" / "out.json").exists()


class TestCompareRuns:
    def test_compare_runs_returns_one_row_per_record(self):
        records = [
            RunRecord.from_summary(_summary(run_id="a"), tags={"ansatz": "hea"}),
            RunRecord.from_summary(_summary(run_id="b")),
        ]
        rows = compare_runs(records)
        assert len(rows) == 2
        assert rows[0]["run_id"] == "a"
        assert json.loads(rows[0]["tags"]) == {"ansatz": "hea"}
        assert rows[1]["tags"] == ""

    def test_compare_runs_empty(self):
        assert compare_runs([]) == []

    def test_format_comparison_table_empty(self):
        assert format_comparison_table([]) == "No runs to compare."

    def test_format_comparison_table_contains_run_ids(self):
        records = [
            RunRecord.from_summary(_summary(run_id="a")),
            RunRecord.from_summary(_summary(run_id="b")),
        ]
        table = format_comparison_table(records)
        assert "a" in table
        assert "b" in table
        assert "run_id" in table  # header present


class TestHistoryReporter:
    def test_wires_into_monitor_and_appends_to_history(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        reporter = HistoryReporter(
            history, tags={"ansatz": "hea"}, framework="pennylane", planned_steps=10
        )
        monitor = QMLMonitor(reporter=reporter, planned_steps=10)
        for step in range(5):
            monitor.update(step=step, loss=1.0 - step * 0.1)
        monitor.finish()

        records = history.load_all()
        assert len(records) == 1
        assert records[0].run_id == monitor.run_id
        assert records[0].framework == "pennylane"
        assert records[0].steps == 5
        assert records[0].tags == {"ansatz": "hea"}

    def test_finalize_returns_underlying_summary(self, tmp_path):
        history = RunHistory(tmp_path / "history.jsonl")
        reporter = HistoryReporter(history)
        monitor = QMLMonitor(reporter=reporter)
        monitor.update(step=0, loss=1.0)
        monitor.finish()
        summary = reporter.finalize()
        assert summary["run_id"] == monitor.run_id
        assert summary["steps"] == 1

    def test_fail_open_when_history_append_raises(self, tmp_path, monkeypatch):
        history = RunHistory(tmp_path / "history.jsonl")

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(history, "append_summary", _boom)
        reporter = HistoryReporter(history)
        monitor = QMLMonitor(reporter=reporter)
        monitor.update(step=0, loss=1.0)
        monitor.finish()  # must not raise
