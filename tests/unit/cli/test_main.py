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
