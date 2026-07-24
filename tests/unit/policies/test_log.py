"""Unit tests for qml_observer.actions.log.LogAction."""

from __future__ import annotations

import logging

from qml_observer.actions.log import LogAction
from qml_observer.schemas.diagnosis import IssueType


class TestLogAction:
    def test_name(self):
        assert LogAction().name == "log"

    def test_always_executes(self, healthy_diagnosis, warning_diagnosis, critical_diagnosis):
        action = LogAction()
        for diagnosis in (healthy_diagnosis, warning_diagnosis, critical_diagnosis):
            result = action.execute(diagnosis)
            assert result.executed is True
            assert result.action_name == "log"

    def test_logs_at_info_for_info_severity(self, healthy_diagnosis, caplog):
        with caplog.at_level(logging.DEBUG, logger="qml_observer.actions"):
            LogAction().execute(healthy_diagnosis)
        assert any(r.levelno == logging.INFO for r in caplog.records)

    def test_logs_at_warning_for_warning_severity(self, warning_diagnosis, caplog):
        with caplog.at_level(logging.DEBUG, logger="qml_observer.actions"):
            LogAction().execute(warning_diagnosis)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_logs_at_error_for_critical_severity(self, critical_diagnosis, caplog):
        with caplog.at_level(logging.DEBUG, logger="qml_observer.actions"):
            LogAction().execute(critical_diagnosis)
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_message_includes_issue_and_confidence(self, warning_diagnosis):
        result = LogAction().execute(warning_diagnosis)
        assert IssueType.STAGNATION.value in result.message

    def test_degraded_flag_reflected_in_log(self, degraded_critical_diagnosis, caplog):
        with caplog.at_level(logging.DEBUG, logger="qml_observer.actions"):
            LogAction().execute(degraded_critical_diagnosis)
        assert any("DEGRADED" in r.message for r in caplog.records)

    def test_custom_logger_is_used(self, healthy_diagnosis):
        custom = logging.getLogger("qml_observer.actions.custom-test")
        custom.setLevel(logging.DEBUG)

        captured = []

        class Capture(logging.Handler):
            def emit(self, record):
                captured.append(record)

        custom.addHandler(Capture())
        LogAction(logger=custom).execute(healthy_diagnosis)
        assert len(captured) == 1

    def test_render_returns_full_explanation(self, warning_diagnosis):
        text = LogAction().render(warning_diagnosis)
        assert "Recommended next steps" in text
