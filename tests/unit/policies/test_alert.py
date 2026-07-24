"""Unit tests for qml_observer.actions.alert.AlertAction."""

from __future__ import annotations

import io
import logging

from qml_observer.actions.alert import AlertAction


class TestAlertAction:
    def test_name(self):
        assert AlertAction().name == "alert"

    def test_skips_info_severity(self, healthy_diagnosis):
        stream = io.StringIO()
        result = AlertAction(stream=stream).execute(healthy_diagnosis)
        assert result.executed is False
        assert stream.getvalue() == ""

    def test_alerts_on_warning_severity(self, warning_diagnosis):
        stream = io.StringIO()
        result = AlertAction(stream=stream).execute(warning_diagnosis)
        assert result.executed is True
        assert "QML OBSERVER ALERT" in stream.getvalue()

    def test_alerts_on_critical_severity(self, critical_diagnosis):
        stream = io.StringIO()
        result = AlertAction(stream=stream).execute(critical_diagnosis)
        assert result.executed is True
        assert "Possible barren plateau" in stream.getvalue()

    def test_logs_warning_level_for_warning_severity(self, warning_diagnosis, caplog):
        stream = io.StringIO()
        with caplog.at_level(logging.DEBUG, logger="qml_observer.actions"):
            AlertAction(stream=stream).execute(warning_diagnosis)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_logs_error_level_for_critical_severity(self, critical_diagnosis, caplog):
        stream = io.StringIO()
        with caplog.at_level(logging.DEBUG, logger="qml_observer.actions"):
            AlertAction(stream=stream).execute(critical_diagnosis)
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_stream_failure_is_reported_not_raised(self, warning_diagnosis):
        class BrokenStream:
            def write(self, *_args, **_kwargs):
                raise OSError("broken pipe")

        result = AlertAction(stream=BrokenStream()).execute(warning_diagnosis)
        assert result.executed is False
        assert "failed" in result.message.lower()

    def test_defaults_to_stderr(self):
        import sys

        assert AlertAction()._stream is sys.stderr
