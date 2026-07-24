"""Unit tests for qml_observer.actions.base."""

from __future__ import annotations

import pytest

from qml_observer.actions.base import Action, ActionResult
from qml_observer.schemas.diagnosis import DiagnosisResult


class TestActionResultValidation:
    def test_minimal_valid_construction(self):
        result = ActionResult(action_name="log", executed=True, message="ok")
        assert result.action_name == "log"
        assert result.executed is True

    def test_empty_action_name_raises(self):
        with pytest.raises(ValueError, match="action_name"):
            ActionResult(action_name="", executed=True, message="ok")

    def test_non_str_action_name_raises(self):
        with pytest.raises(TypeError, match="action_name"):
            ActionResult(action_name=5, executed=True, message="ok")

    def test_non_bool_executed_raises(self):
        with pytest.raises(TypeError, match="executed"):
            ActionResult(action_name="log", executed="yes", message="ok")

    def test_non_str_message_raises(self):
        with pytest.raises(TypeError, match="message"):
            ActionResult(action_name="log", executed=True, message=123)


class TestActionIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Action()  # type: ignore[abstract]

    def test_subclass_must_implement_execute(self):
        class Incomplete(Action):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self, healthy_diagnosis):
        class AlwaysLogs(Action):
            name = "always-logs"

            def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
                return ActionResult(action_name=self.name, executed=True, message="did it")

        action = AlwaysLogs()
        result = action.execute(healthy_diagnosis)
        assert result.executed is True
        assert result.action_name == "always-logs"
