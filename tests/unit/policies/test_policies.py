"""Unit tests for qml_observer.actions.policies.ActionPolicy."""

from __future__ import annotations

import pytest

from qml_observer.actions.alert import AlertAction
from qml_observer.actions.log import LogAction
from qml_observer.actions.pause import PauseAction
from qml_observer.actions.policies import VALID_MODES, ActionPolicy
from qml_observer.actions.stop import StopAction


class TestConstruction:
    def test_default_mode_is_warn(self):
        assert ActionPolicy().mode == "warn"

    @pytest.mark.parametrize("mode", sorted(VALID_MODES))
    def test_all_valid_modes_accepted(self, mode):
        assert ActionPolicy(mode=mode).mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            ActionPolicy(mode="bogus")

    def test_default_allow_stop_on_degraded_is_false(self):
        assert ActionPolicy(mode="adaptive").allow_stop_on_degraded is False


class TestLogMode:
    def test_always_selects_log_action(
        self, healthy_diagnosis, warning_diagnosis, critical_diagnosis
    ):
        policy = ActionPolicy(mode="log")
        for diagnosis in (healthy_diagnosis, warning_diagnosis, critical_diagnosis):
            assert isinstance(policy.select_action(diagnosis), LogAction)


class TestWarnMode:
    def test_info_selects_log_action(self, healthy_diagnosis):
        assert isinstance(ActionPolicy(mode="warn").select_action(healthy_diagnosis), LogAction)

    def test_warning_selects_alert_action(self, warning_diagnosis):
        assert isinstance(ActionPolicy(mode="warn").select_action(warning_diagnosis), AlertAction)

    def test_critical_never_stops_in_warn_mode(self, critical_diagnosis):
        assert isinstance(ActionPolicy(mode="warn").select_action(critical_diagnosis), AlertAction)


class TestPauseMode:
    """`pause` selects a real `PauseAction` for critical diagnoses (Issue #90b)."""

    def test_critical_selects_pause_action(self, critical_diagnosis):
        action = ActionPolicy(mode="pause").select_action(critical_diagnosis)
        assert isinstance(action, PauseAction)
        assert not isinstance(action, StopAction)

    def test_critical_never_stops_in_pause_mode(self, critical_diagnosis):
        action = ActionPolicy(mode="pause").select_action(critical_diagnosis)
        assert not isinstance(action, StopAction)

    def test_warning_selects_alert_not_pause(self, warning_diagnosis):
        action = ActionPolicy(mode="pause").select_action(warning_diagnosis)
        assert isinstance(action, AlertAction)

    def test_info_selects_log_action(self, healthy_diagnosis):
        assert isinstance(ActionPolicy(mode="pause").select_action(healthy_diagnosis), LogAction)

    def test_degraded_critical_never_pauses(self, degraded_critical_diagnosis):
        """Addendum §1 applies to PauseAction too: degraded never escalates."""
        action = ActionPolicy(mode="pause").select_action(degraded_critical_diagnosis)
        assert isinstance(action, AlertAction)

    def test_stop_mode_never_selects_pause(self, critical_diagnosis):
        """`stop`/`adaptive` escalate straight past pausing (module docstring)."""
        action = ActionPolicy(mode="stop").select_action(critical_diagnosis)
        assert isinstance(action, StopAction)
        assert not isinstance(action, PauseAction)


class TestStopMode:
    def test_critical_selects_stop_action(self, critical_diagnosis):
        assert isinstance(ActionPolicy(mode="stop").select_action(critical_diagnosis), StopAction)

    def test_warning_selects_alert_not_stop(self, warning_diagnosis):
        assert isinstance(ActionPolicy(mode="stop").select_action(warning_diagnosis), AlertAction)

    def test_info_selects_log_action(self, healthy_diagnosis):
        assert isinstance(ActionPolicy(mode="stop").select_action(healthy_diagnosis), LogAction)

    def test_degraded_critical_never_stops(self, degraded_critical_diagnosis):
        """Addendum §1: a degraded diagnosis never escalates to StopAction."""
        action = ActionPolicy(mode="stop").select_action(degraded_critical_diagnosis)
        assert isinstance(action, AlertAction)

    def test_execute_actually_triggers_stop_action(self, critical_diagnosis):
        stop_action = StopAction()
        policy = ActionPolicy(mode="stop", stop_action=stop_action)
        result = policy.execute(critical_diagnosis)
        assert result.action_name == "stop"
        assert stop_action.triggered is True

    def test_reset_clears_stop_action(self, critical_diagnosis):
        stop_action = StopAction()
        policy = ActionPolicy(mode="stop", stop_action=stop_action)
        policy.execute(critical_diagnosis)
        assert stop_action.triggered is True
        policy.reset()
        assert stop_action.triggered is False


class TestAdaptiveMode:
    def test_behaves_like_stop_when_not_degraded(self, critical_diagnosis):
        action = ActionPolicy(mode="adaptive").select_action(critical_diagnosis)
        assert isinstance(action, StopAction)

    def test_degraded_critical_does_not_stop_by_default(self, degraded_critical_diagnosis):
        action = ActionPolicy(mode="adaptive").select_action(degraded_critical_diagnosis)
        assert isinstance(action, AlertAction)

    def test_degraded_critical_stops_with_explicit_opt_in(self, degraded_critical_diagnosis):
        policy = ActionPolicy(mode="adaptive", allow_stop_on_degraded=True)
        action = policy.select_action(degraded_critical_diagnosis)
        assert isinstance(action, StopAction)

    def test_opt_in_flag_ignored_for_other_modes(self, degraded_critical_diagnosis):
        """`allow_stop_on_degraded` only matters combined with mode='adaptive'."""
        policy = ActionPolicy(mode="stop", allow_stop_on_degraded=True)
        action = policy.select_action(degraded_critical_diagnosis)
        assert isinstance(action, AlertAction)


class TestExecute:
    def test_execute_returns_action_result(self, warning_diagnosis):
        result = ActionPolicy(mode="warn").execute(warning_diagnosis)
        assert result.action_name == "alert"
        assert result.executed is True

    def test_custom_action_instances_are_used(self, warning_diagnosis):
        log_action = LogAction()
        alert_action = AlertAction()
        policy = ActionPolicy(mode="warn", log_action=log_action, alert_action=alert_action)
        assert policy.select_action(warning_diagnosis) is alert_action
