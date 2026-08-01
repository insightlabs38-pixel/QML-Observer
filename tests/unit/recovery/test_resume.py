"""Unit tests for qml_observer.recovery.resume.resume_monitor_from_snapshot."""

from __future__ import annotations

import numpy as np
import pytest

from qml_observer.actions.pause import PausedRunSnapshot
from qml_observer.actions.policies import ActionPolicy
from qml_observer.core.monitor import QMLMonitor
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.recovery.resume import resume_monitor_from_snapshot
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType


def _snapshot(**overrides) -> PausedRunSnapshot:
    diagnosis = DiagnosisResult(
        issue=IssueType.POSSIBLE_BARREN_PLATEAU,
        confidence=0.9,
        severity="critical",
        evidence=["synthetic"],
        recommendations=["synthetic"],
    )
    defaults = dict(
        run_id="paused-run",
        step=42,
        paused_at=1234.0,
        diagnosis=diagnosis,
        window_size=50,
        planned_steps=500,
        extra={},
    )
    defaults.update(overrides)
    return PausedRunSnapshot(**defaults)


class TestValidation:
    def test_rejects_non_snapshot(self):
        with pytest.raises(TypeError, match="PausedRunSnapshot"):
            resume_monitor_from_snapshot("not-a-snapshot")  # type: ignore[arg-type]


class TestReconstruction:
    def test_preserves_run_id(self):
        monitor = resume_monitor_from_snapshot(_snapshot(run_id="my-run"))
        assert monitor.run_id == "my-run"

    def test_preserves_window_size(self):
        monitor = resume_monitor_from_snapshot(_snapshot(window_size=77))
        assert monitor.state.window_size == 77

    def test_preserves_planned_steps(self):
        monitor = resume_monitor_from_snapshot(_snapshot(planned_steps=999))
        assert monitor._planned_steps == 999

    def test_seeds_step_count(self):
        monitor = resume_monitor_from_snapshot(_snapshot(step=123))
        assert monitor.state.step_count == 123

    def test_window_starts_empty(self):
        monitor = resume_monitor_from_snapshot(_snapshot())
        assert monitor.state.window == []

    def test_default_policy_is_pause(self):
        monitor = resume_monitor_from_snapshot(_snapshot())
        assert monitor.policy == "pause"

    def test_custom_policy(self):
        monitor = resume_monitor_from_snapshot(_snapshot(), policy="warn")
        assert monitor.policy == "warn"

    def test_custom_action_policy(self):
        custom = ActionPolicy(mode="stop")
        monitor = resume_monitor_from_snapshot(_snapshot(), action_policy=custom)
        assert monitor.action_policy is custom

    def test_detectors_are_passed_through(self):
        detector = BarrenPlateauDetector()
        monitor = resume_monitor_from_snapshot(_snapshot(), detectors=[detector])
        assert monitor._detectors == [detector]

    def test_fallback_window_size_when_zero(self):
        # A snapshot captured without run context (PauseAction.execute(diagnosis)
        # with no keyword args) has window_size=0; resuming should not construct
        # an invalid (window_size < 1) monitor.
        monitor = resume_monitor_from_snapshot(_snapshot(window_size=0))
        assert monitor.state.window_size == 100

    def test_fallback_run_id_when_unknown(self):
        monitor = resume_monitor_from_snapshot(_snapshot(run_id="unknown"))
        assert monitor.run_id != "unknown"  # a fresh run_id was generated instead


class TestEndToEndResume:
    def test_resumed_monitor_continues_step_sequence(self):
        original = QMLMonitor(
            detectors=[
                BarrenPlateauDetector(
                    gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=5
                )
            ],
            policy="pause",
            window_size=50,
            planned_steps=200,
            run_id="e2e-run",
        )
        rng = np.random.default_rng(0)
        for step in range(10):
            original.update(
                step=step, loss=0.8 + rng.normal(0, 1e-8), gradients=rng.normal(0, 1e-6, size=8)
            )
        assert original.should_pause() is True
        snapshot = original.action_policy.pause_action.last_snapshot
        assert snapshot is not None

        resumed = resume_monitor_from_snapshot(
            snapshot,
            detectors=[
                BarrenPlateauDetector(
                    gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=5
                )
            ],
        )
        assert resumed.run_id == "e2e-run"
        assert resumed.state.step_count == snapshot.step
        assert resumed._planned_steps == 200
        assert resumed.state.window_size == 50

        # Feed it healthy-looking data post-resume and confirm it behaves
        # like a normal, functioning monitor continuing the step sequence.
        result = resumed.update(step=snapshot.step, loss=0.5, gradients=rng.normal(0, 0.5, size=8))
        assert resumed.state.step_count == snapshot.step + 1
        assert isinstance(result, DiagnosisResult)
