"""Unit tests for qml_observer.actions.pause.PauseAction (Issue #90b)."""

from __future__ import annotations

from qml_observer.actions.pause import PauseAction, PausedRunSnapshot


class TestPauseAction:
    def test_name(self):
        assert PauseAction().name == "pause"

    def test_not_triggered_initially(self):
        action = PauseAction()
        assert action.triggered is False
        assert action.paused is False
        assert action.last_diagnosis is None
        assert action.last_snapshot is None

    def test_execute_sets_triggered_and_paused(self, critical_diagnosis):
        action = PauseAction()
        result = action.execute(critical_diagnosis)
        assert action.triggered is True
        assert action.paused is True
        assert action.last_diagnosis is critical_diagnosis
        assert result.executed is True
        assert result.action_name == "pause"

    def test_execute_captures_snapshot_with_run_context(self, critical_diagnosis):
        action = PauseAction()
        action.execute(
            critical_diagnosis,
            run_id="run-123",
            step=42,
            window_size=100,
            planned_steps=1000,
            extra={"optimizer_checkpoint": "/tmp/ckpt.npz"},
        )
        snapshot = action.last_snapshot
        assert isinstance(snapshot, PausedRunSnapshot)
        assert snapshot.run_id == "run-123"
        assert snapshot.step == 42
        assert snapshot.window_size == 100
        assert snapshot.planned_steps == 1000
        assert snapshot.diagnosis is critical_diagnosis
        assert snapshot.extra == {"optimizer_checkpoint": "/tmp/ckpt.npz"}
        assert snapshot.paused_at > 0

    def test_execute_without_context_still_captures_a_snapshot(self, critical_diagnosis):
        """The bare `Action.execute(diagnosis)` contract must still work."""
        action = PauseAction()
        action.execute(critical_diagnosis)
        assert action.last_snapshot is not None
        assert action.last_snapshot.run_id == "unknown"
        assert action.last_snapshot.step == 0

    def test_resume_clears_triggered_but_keeps_history(self, critical_diagnosis):
        action = PauseAction()
        action.execute(critical_diagnosis, run_id="run-1", step=5)
        action.resume()
        assert action.triggered is False
        assert action.paused is False
        # History is preserved for inspection/audit after a resume.
        assert action.last_diagnosis is critical_diagnosis
        assert action.last_snapshot is not None
        assert action.last_snapshot.run_id == "run-1"

    def test_reset_clears_triggered_and_all_history(self, critical_diagnosis):
        action = PauseAction()
        action.execute(critical_diagnosis, run_id="run-1", step=5)
        action.reset()
        assert action.triggered is False
        assert action.last_diagnosis is None
        assert action.last_snapshot is None

    def test_message_explains_caller_responsibility(self, critical_diagnosis):
        result = PauseAction().execute(critical_diagnosis)
        assert "training loop" in result.message
        assert "last_snapshot" in result.message or "should_pause" in result.message

    def test_logging_failure_does_not_suppress_the_pause(self, critical_diagnosis):
        class BrokenLogger:
            def warning(self, *_args, **_kwargs):
                raise RuntimeError("logging backend down")

        action = PauseAction(logger=BrokenLogger())
        result = action.execute(critical_diagnosis)
        assert action.triggered is True  # the pause is still recorded
        assert action.last_snapshot is not None
        assert result.executed is True
        assert "logging the pause failed" in result.message

    def test_can_pause_and_resume_multiple_times(self, critical_diagnosis, healthy_diagnosis):
        action = PauseAction()
        action.execute(critical_diagnosis, run_id="run-1", step=1)
        action.resume()
        assert action.triggered is False
        action.execute(critical_diagnosis, run_id="run-1", step=2)
        assert action.triggered is True
        assert action.last_snapshot.step == 2
