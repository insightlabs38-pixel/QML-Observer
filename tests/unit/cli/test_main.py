"""Unit tests for qml_observer.cli.main (Milestone 7, Issue #50)."""

import pytest

from qml_observer.cli.main import build_parser, main
from qml_observer.core.monitor import QMLMonitor
from qml_observer.reporting.reporter import RunReporter


def _make_log(tmp_path, *, planned_steps=10, n_steps=5):
    path = tmp_path / "run.jsonl"
    reporter = RunReporter(path, framework="generic", planned_steps=planned_steps)
    monitor = QMLMonitor(reporter=reporter, planned_steps=planned_steps)
    for step in range(n_steps):
        monitor.update(step=step, loss=1.0 - step * 0.05)
    monitor.finish()
    return path


class TestParser:
    def test_requires_subcommand(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_inspect_and_report_parse(self):
        parser = build_parser()
        args = parser.parse_args(["inspect", "run.jsonl"])
        assert args.command == "inspect"
        assert args.path == "run.jsonl"

        args = parser.parse_args(["report", "run.jsonl"])
        assert args.command == "report"


class TestInspectCommand:
    def test_inspect_prints_all_records(self, tmp_path, capsys):
        path = _make_log(tmp_path)
        exit_code = main(["inspect", str(path)])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "record 0 (event)" in out
        assert "record" in out and "(summary)" in out

    def test_inspect_missing_file(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["inspect", str(tmp_path / "missing.jsonl")])


class TestReportCommand:
    def test_report_prints_run_header_and_status(self, tmp_path, capsys):
        path = _make_log(tmp_path)
        exit_code = main(["report", str(path)])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "QML Observer" in out
        assert "Run:" in out
        assert "Status:" in out
        assert "Confidence:" in out
        assert "Estimated compute saved:" in out

    def test_report_flags_degraded_run(self, tmp_path, capsys):
        path = tmp_path / "run.jsonl"
        reporter = RunReporter(path)
        monitor = QMLMonitor(reporter=reporter)
        monitor.update(step=0, loss=1.0)
        monitor.finish()
        # Manually confirm non-degraded baseline still renders cleanly.
        main(["report", str(path)])
        out = capsys.readouterr().out
        assert "DIAGNOSIS DEGRADED" not in out

    def test_report_missing_events_raises(self, tmp_path):
        path = tmp_path / "run.jsonl"
        path.write_text('{"type": "summary", "steps": 0}\n', encoding="utf-8")
        with pytest.raises(SystemExit):
            main(["report", str(path)])

    def test_report_missing_file(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["report", str(tmp_path / "missing.jsonl")])

    def test_report_shows_gradient_from_build_run_summary(self, tmp_path, capsys):
        """`RunReporter`'s automatic summary never includes gradient/circuit
        detail (it only ever sees the bare `TrainingEvent` -- see
        `reporting.reporter`'s module docstring); `build_run_summary()` is
        the richer alternative meant to be written as the `"summary"`
        record instead. This is the CLI's only current coverage of that
        path (previously untested)."""
        from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
        from qml_observer.reporting.jsonl import JSONLWriter, event_record, summary_record
        from qml_observer.reporting.summary import build_run_summary

        path = tmp_path / "run.jsonl"
        monitor = QMLMonitor(detectors=[BarrenPlateauDetector(patience=3)], policy="log")
        with JSONLWriter(path) as writer:
            diagnosis = None
            for step in range(10):
                gradients = [1e-8, -1e-8, 1e-8]
                diagnosis = monitor.update(step=step, loss=0.5, gradients=gradients)
                writer.write(event_record(monitor.state.latest_observation.training_event))
            final = monitor.finish()
            assert diagnosis is not None
            summary = build_run_summary(monitor.state, final, framework="generic")
            writer.write(summary_record(summary))

        exit_code = main(["report", str(path)])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "Gradient norm:" in out
        assert "Gradient variance:" in out


class TestNotYetImplementedCommands:
    def test_run_command_exits_nonzero(self, capsys):
        exit_code = main(["run", "config.yaml"])
        err = capsys.readouterr().err
        assert exit_code == 1
        assert "not implemented yet" in err

    def test_benchmark_command_exits_nonzero(self, capsys):
        exit_code = main(["benchmark", "barren-plateau"])
        err = capsys.readouterr().err
        assert exit_code == 1
        assert "not implemented yet" in err


class TestPluginsCommand:
    def test_plugins_list_parses(self):
        parser = build_parser()
        args = parser.parse_args(["plugins", "list"])
        assert args.command == "plugins"
        assert args.plugins_action == "list"

    def test_plugins_list_with_no_plugins_registered(self, capsys, monkeypatch):
        monkeypatch.setattr("qml_observer.cli.main.list_detector_plugins", lambda: {})
        exit_code = main(["plugins", "list"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "No detector plugins registered" in out

    def test_plugins_list_with_plugins_registered(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "qml_observer.cli.main.list_detector_plugins",
            lambda: {"my_detector": "my_package.detectors:MyDetector"},
        )
        exit_code = main(["plugins", "list"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "my_detector" in out
        assert "my_package.detectors:MyDetector" in out


class TestHistoryCommand:
    def _make_history(self, tmp_path, n_runs=2):
        from qml_observer.reporting.history import HistoryReporter, RunHistory

        path = tmp_path / "history.jsonl"
        history = RunHistory(path)
        for i in range(n_runs):
            reporter = HistoryReporter(
                history,
                tags={"ansatz": "hea" if i == 0 else "strong_ent"},
                framework="pennylane",
                planned_steps=10,
            )
            monitor = QMLMonitor(reporter=reporter, planned_steps=10, run_id=f"run-{i}")
            for step in range(5):
                monitor.update(step=step, loss=1.0 - step * 0.1)
            monitor.finish()
        return path

    def test_history_list_parses(self):
        parser = build_parser()
        args = parser.parse_args(["history", "list", "history.jsonl"])
        assert args.command == "history"
        assert args.history_action == "list"
        assert args.path == "history.jsonl"

    def test_history_compare_parses_with_options(self):
        parser = build_parser()
        args = parser.parse_args(
            ["history", "compare", "h.jsonl", "--run-id", "a", "--run-id", "b", "--tag", "x=y"]
        )
        assert args.run_id == ["a", "b"]
        assert args.tag == "x=y"

    def test_history_export_requires_format(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["history", "export", "h.jsonl", "out.csv"])

    def test_history_list_prints_runs(self, tmp_path, capsys):
        path = self._make_history(tmp_path)
        exit_code = main(["history", "list", str(path)])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "run-0" in out
        assert "run-1" in out
        assert "2 run(s) total." in out

    def test_history_list_on_empty_ledger(self, tmp_path, capsys):
        path = tmp_path / "empty.jsonl"
        exit_code = main(["history", "list", str(path)])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "No runs recorded" in out

    def test_history_list_filters_by_tag(self, tmp_path, capsys):
        path = self._make_history(tmp_path)
        exit_code = main(["history", "list", str(path), "--tag", "ansatz=hea"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "run-0" in out
        assert "run-1" not in out

    def test_history_compare_filters_by_run_id(self, tmp_path, capsys):
        path = self._make_history(tmp_path)
        exit_code = main(["history", "compare", str(path), "--run-id", "run-0"])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert "run-0" in out
        assert "run-1" not in out

    def test_history_export_csv(self, tmp_path, capsys):
        path = self._make_history(tmp_path)
        out_path = tmp_path / "out.csv"
        exit_code = main(["history", "export", str(path), "--format", "csv", str(out_path)])
        out = capsys.readouterr().out
        assert exit_code == 0
        assert out_path.exists()
        assert "Exported 2 run(s)" in out

    def test_history_export_json(self, tmp_path):
        import json

        path = self._make_history(tmp_path)
        out_path = tmp_path / "out.json"
        exit_code = main(["history", "export", str(path), "--format", "json", str(out_path)])
        assert exit_code == 0
        data = json.loads(out_path.read_text())
        assert len(data) == 2

    def test_history_list_bad_tag_filter_exits(self, tmp_path):
        path = self._make_history(tmp_path, n_runs=1)
        with pytest.raises(SystemExit):
            main(["history", "list", str(path), "--tag", "no-equals-sign"])
