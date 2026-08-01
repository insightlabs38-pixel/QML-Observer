"""QMLMonitor <-> ActionPolicy integration tests.

Milestone 5, Issue #38 ("Add warn mode"), Issue #39 ("Add stop mode"),
Issue #40 ("Test action safety").
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from qml_observer.actions.base import Action, ActionResult
from qml_observer.actions.policies import ActionPolicy
from qml_observer.actions.stop import StopAction
from qml_observer.core.monitor import QMLMonitor
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType


def _critical_diagnosis() -> DiagnosisResult:
    return DiagnosisResult(
        issue=IssueType.POSSIBLE_BARREN_PLATEAU,
        confidence=0.9,
        severity="critical",
        evidence=["synthetic"],
        recommendations=["synthetic"],
    )


class TestActionPolicyWiring:
    def test_default_action_policy_matches_policy_string(self):
        monitor = QMLMonitor(policy="stop")
        assert isinstance(monitor.action_policy, ActionPolicy)
        assert monitor.action_policy.mode == "stop"

    def test_custom_action_policy_overrides_policy_string(self):
        custom = ActionPolicy(mode="adaptive", allow_stop_on_degraded=True)
        monitor = QMLMonitor(action_policy=custom)
        assert monitor.policy == "adaptive"
        assert monitor.action_policy is custom

    def test_invalid_action_policy_type_raises(self):
        with pytest.raises(TypeError, match="ActionPolicy"):
            QMLMonitor(action_policy="not-a-policy")  # type: ignore[arg-type]


class TestWarnMode:
    def test_warn_mode_never_stops_even_for_critical(self):
        monitor = QMLMonitor(policy="warn")
        monitor.state.latest_diagnosis = _critical_diagnosis()
        assert monitor.should_stop() is False

    def test_update_runs_action_policy_and_exposes_result(self):
        monitor = QMLMonitor(policy="warn")
        monitor.update(step=0, loss=1.0)  # default diagnosis: severity "info"
        result = monitor.latest_action_result()
        assert isinstance(result, ActionResult)
        assert result.action_name == "log"  # info severity -> quiet log, not alert

    def test_finish_also_runs_action_policy(self):
        monitor = QMLMonitor(policy="warn")
        monitor.update(step=0, loss=1.0)
        monitor.finish()
        assert monitor.latest_action_result() is not None


class TestStopMode:
    def test_real_barren_plateau_run_arms_stop_action(self):
        monitor = QMLMonitor(
            detectors=[
                BarrenPlateauDetector(
                    gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=10
                )
            ],
            policy="stop",
        )
        rng = np.random.default_rng(0)
        for step in range(20):
            monitor.update(
                step=step, loss=0.8 + rng.normal(0, 1e-8), gradients=rng.normal(0, 1e-6, size=8)
            )
        assert monitor.should_stop() is True
        # execute() ran as part of update(), so the StopAction itself was triggered too.
        assert monitor.action_policy.stop_action.triggered is True
        result = monitor.latest_action_result()
        assert result is not None
        assert result.action_name == "stop"

    def test_reset_clears_stop_action_state(self):
        monitor = QMLMonitor(policy="stop")
        # Arm the StopAction directly (update() with no detectors would
        # otherwise overwrite state.latest_diagnosis with the "info"
        # placeholder before the action policy runs).
        monitor.action_policy.execute(_critical_diagnosis())
        assert monitor.action_policy.stop_action.triggered is True
        monitor.reset()
        assert monitor.action_policy.stop_action.triggered is False
        assert monitor.should_stop() is False

    def test_adaptive_mode_stops_on_non_degraded_critical(self):
        monitor = QMLMonitor(policy="adaptive")
        monitor.state.latest_diagnosis = _critical_diagnosis()
        assert monitor.should_stop() is True

    def test_adaptive_mode_does_not_stop_on_degraded_by_default(self):
        monitor = QMLMonitor(policy="adaptive")
        monitor.state.latest_diagnosis = DiagnosisResult(
            issue=IssueType.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            severity="critical",
            evidence=[],
            recommendations=[],
            degraded=True,
            degraded_reason="synthetic",
        )
        assert monitor.should_stop() is False


class TestPauseMode:
    """Milestone 13, Issue #90b: `PauseAction` real behavior via `QMLMonitor`."""

    def test_real_barren_plateau_run_arms_pause_action(self):
        monitor = QMLMonitor(
            detectors=[
                BarrenPlateauDetector(
                    gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=10
                )
            ],
            policy="pause",
        )
        rng = np.random.default_rng(0)
        for step in range(20):
            monitor.update(
                step=step, loss=0.8 + rng.normal(0, 1e-8), gradients=rng.normal(0, 1e-6, size=8)
            )
        assert monitor.should_pause() is True
        assert monitor.should_stop() is False  # pause never escalates to a stop by itself
        assert monitor.action_policy.pause_action.triggered is True
        result = monitor.latest_action_result()
        assert result is not None
        assert result.action_name == "pause"

    def test_pause_snapshot_captures_run_context(self):
        monitor = QMLMonitor(policy="pause", window_size=50, planned_steps=500, run_id="paused-1")
        monitor.state.latest_diagnosis = _critical_diagnosis()
        monitor.action_policy.execute(
            _critical_diagnosis(),
            run_id=monitor.run_id,
            step=monitor.state.step_count,
            window_size=50,
            planned_steps=500,
        )
        snapshot = monitor.action_policy.pause_action.last_snapshot
        assert snapshot is not None
        assert snapshot.run_id == "paused-1"
        assert snapshot.window_size == 50
        assert snapshot.planned_steps == 500

    def test_update_populates_snapshot_via_monitor_run_context(self):
        monitor = QMLMonitor(
            detectors=[
                BarrenPlateauDetector(
                    gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=5
                )
            ],
            policy="pause",
            window_size=100,
            run_id="paused-run-ctx",
        )
        rng = np.random.default_rng(1)
        for step in range(10):
            monitor.update(
                step=step, loss=0.8 + rng.normal(0, 1e-8), gradients=rng.normal(0, 1e-6, size=8)
            )
        snapshot = monitor.action_policy.pause_action.last_snapshot
        assert snapshot is not None
        assert snapshot.run_id == "paused-run-ctx"
        assert snapshot.window_size == 100

    def test_reset_clears_pause_action_state(self):
        monitor = QMLMonitor(policy="pause")
        monitor.action_policy.execute(_critical_diagnosis())
        assert monitor.action_policy.pause_action.triggered is True
        monitor.reset()
        assert monitor.action_policy.pause_action.triggered is False
        assert monitor.should_pause() is False

    def test_degraded_critical_never_pauses(self):
        monitor = QMLMonitor(policy="pause")
        monitor.state.latest_diagnosis = DiagnosisResult(
            issue=IssueType.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            severity="critical",
            evidence=[],
            recommendations=[],
            degraded=True,
            degraded_reason="synthetic",
        )
        assert monitor.should_pause() is False


class TestActionSafety:
    """Issue #40: an action-layer failure must never propagate into the caller."""

    def test_broken_custom_action_does_not_crash_update(self, caplog):
        class BrokenAction(Action):
            name = "broken"

            def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
                raise RuntimeError("this action is broken")

        policy = ActionPolicy(mode="log", log_action=BrokenAction())
        monitor = QMLMonitor(action_policy=policy)

        with caplog.at_level(logging.WARNING, logger="qml_observer"):
            diagnosis = monitor.update(step=0, loss=1.0)  # must not raise

        assert diagnosis is not None
        assert monitor.latest_action_result() is None
        assert any("action policy failed" in r.message for r in caplog.records)

    def test_broken_custom_action_does_not_crash_finish(self, caplog):
        class BrokenAction(Action):
            name = "broken"

            def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
                raise RuntimeError("this action is broken")

        policy = ActionPolicy(mode="log", log_action=BrokenAction())
        monitor = QMLMonitor(action_policy=policy)
        monitor.update(step=0, loss=1.0)

        with caplog.at_level(logging.WARNING, logger="qml_observer"):
            diagnosis = monitor.finish()  # must not raise

        assert diagnosis is not None

    def test_training_loop_continues_after_repeated_action_failures(self):
        class AlwaysBroken(Action):
            name = "broken"

            def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
                raise ValueError("nope")

        policy = ActionPolicy(mode="log", log_action=AlwaysBroken())
        monitor = QMLMonitor(action_policy=policy)
        for step in range(10):
            monitor.update(step=step, loss=1.0 - step * 0.01)
        assert monitor.state.step_count == 10  # the loop ran to completion

    def test_builtin_actions_never_raise_even_when_misconfigured(self):
        """Sanity check that the built-in stop-mode path itself is safe end to end."""
        stop_action = StopAction()
        policy = ActionPolicy(mode="stop", stop_action=stop_action)
        monitor = QMLMonitor(action_policy=policy)
        monitor.state.latest_diagnosis = _critical_diagnosis()
        monitor.update(step=0, loss=1.0)  # must not raise
        result = monitor.latest_action_result()
        assert result is not None
