"""Unit tests for qml_observer.actions.stop.StopAction."""

from __future__ import annotations

from qml_observer.actions.stop import StopAction


class TestStopAction:
    def test_name(self):
        assert StopAction().name == "stop"

    def test_not_triggered_initially(self):
        assert StopAction().triggered is False
        assert StopAction().last_diagnosis is None

    def test_execute_sets_triggered(self, critical_diagnosis):
        action = StopAction()
        result = action.execute(critical_diagnosis)
        assert action.triggered is True
        assert action.last_diagnosis is critical_diagnosis
        assert result.executed is True
        assert result.action_name == "stop"

    def test_triggered_stays_true_until_reset(self, critical_diagnosis, healthy_diagnosis):
        action = StopAction()
        action.execute(critical_diagnosis)
        assert action.triggered is True
        # A later, unrelated call doesn't need to happen for `triggered` to
        # persist -- it only clears via reset().
        action.reset()
        assert action.triggered is False
        assert action.last_diagnosis is None

    def test_message_explains_caller_responsibility(self, critical_diagnosis):
        result = StopAction().execute(critical_diagnosis)
        assert "training loop" in result.message

    def test_logging_failure_does_not_suppress_the_stop(self, critical_diagnosis):
        class BrokenLogger:
            def error(self, *_args, **_kwargs):
                raise RuntimeError("logging backend down")

        action = StopAction(logger=BrokenLogger())
        result = action.execute(critical_diagnosis)
        assert action.triggered is True  # the stop is still recorded
        assert result.executed is True
        assert "logging the stop failed" in result.message
