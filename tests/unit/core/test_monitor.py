"""Unit tests for qml_observer.core.monitor.QMLMonitor."""

import time

import numpy as np
import pytest

from qml_observer.core.monitor import QMLMonitor
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType


class TestConstruction:
    def test_defaults(self):
        monitor = QMLMonitor()
        assert monitor.run_id.startswith("run-")
        assert monitor.policy == "warn"
        assert monitor.state.window_size == 100
        assert monitor.latest_diagnosis() is None

    def test_custom_run_id(self):
        monitor = QMLMonitor(run_id="my-experiment")
        assert monitor.run_id == "my-experiment"

    def test_invalid_policy_raises(self):
        with pytest.raises(ValueError):
            QMLMonitor(policy="nonsense")

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError):
            QMLMonitor(window_size=0)

    def test_invalid_planned_steps_raises(self):
        with pytest.raises(ValueError):
            QMLMonitor(planned_steps=-5)

    def test_empty_run_id_raises(self):
        with pytest.raises(ValueError):
            QMLMonitor(run_id="")


class TestLifecycleBasics:
    def test_start_is_idempotent(self):
        monitor = QMLMonitor()
        monitor.start()
        start_time = monitor.state.start_time
        monitor.start()
        assert monitor.state.start_time == start_time

    def test_update_auto_starts(self):
        monitor = QMLMonitor()
        assert monitor.state.started is False
        monitor.update(step=0, loss=1.0)
        assert monitor.state.started is True

    def test_update_returns_diagnosis_result(self):
        monitor = QMLMonitor()
        diagnosis = monitor.update(step=0, loss=1.0)
        assert isinstance(diagnosis, DiagnosisResult)
        assert diagnosis.issue == IssueType.INSUFFICIENT_EVIDENCE
        assert diagnosis.degraded is False

    def test_update_records_observation(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0)
        monitor.update(step=1, loss=0.9)
        assert monitor.state.step_count == 2
        assert monitor.state.latest_observation.training_event.step == 1
        assert monitor.state.latest_observation.training_event.loss == 0.9

    def test_update_computes_wall_time_between_calls(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0)
        time.sleep(0.01)
        monitor.update(step=1, loss=0.9)
        wall_time = monitor.state.latest_observation.training_event.wall_time
        assert wall_time is not None
        assert wall_time >= 0.005

    def test_first_update_wall_time_measured_from_start(self):
        # start() (called implicitly by update()) begins wall-time tracking,
        # so even the first update has a (small, non-negative) wall_time
        # measured from start() rather than None.
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0)
        wall_time = monitor.state.latest_observation.training_event.wall_time
        assert wall_time is not None
        assert wall_time >= 0.0

    def test_wall_time_none_only_before_any_start(self):
        monitor = QMLMonitor()
        assert monitor.state.start_time is None
        assert monitor._last_perf_time is None

    def test_update_with_gradients_summarizes(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0, gradients=np.array([0.1, -0.2, 0.3]))
        grad = monitor.state.latest_observation.gradient
        assert grad is not None
        assert grad.norm_l2 == pytest.approx(np.linalg.norm([0.1, -0.2, 0.3]))

    def test_update_after_finish_raises(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0)
        monitor.finish()
        with pytest.raises(RuntimeError):
            monitor.update(step=1, loss=0.9)

    def test_finish_without_start_raises(self):
        monitor = QMLMonitor()
        with pytest.raises(RuntimeError):
            monitor.finish()

    def test_finish_returns_diagnosis(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0)
        diagnosis = monitor.finish()
        assert isinstance(diagnosis, DiagnosisResult)
        assert monitor.state.finished is True
        assert monitor.state.end_time is not None

    def test_finish_is_idempotent(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0)
        first = monitor.finish()
        second = monitor.finish()
        assert first is second


class TestFailOpen:
    def test_empty_gradient_array_degrades_without_raising(self):
        monitor = QMLMonitor()
        diagnosis = monitor.update(step=0, loss=1.0, gradients=np.array([]))
        assert diagnosis.degraded is True
        assert diagnosis.degraded_reason is not None
        assert "ValueError" in diagnosis.degraded_reason

    def test_degraded_step_does_not_stop_the_loop(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0, gradients=np.array([]))
        # The monitor itself did not raise, so a subsequent update works fine.
        diagnosis = monitor.update(step=1, loss=0.9)
        assert diagnosis.degraded is False

    def test_invalid_step_type_degrades(self):
        monitor = QMLMonitor()
        diagnosis = monitor.update(step="not-an-int", loss=1.0)
        assert diagnosis.degraded is True


class TestReset:
    def test_reset_clears_state(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0)
        monitor.finish()
        old_run_id = monitor.run_id

        monitor.reset()

        assert monitor.run_id != old_run_id
        assert monitor.state.step_count == 0
        assert monitor.state.started is False
        assert monitor.state.finished is False
        assert monitor.latest_diagnosis() is None

    def test_reset_with_explicit_run_id(self):
        monitor = QMLMonitor()
        monitor.reset(run_id="new-run")
        assert monitor.run_id == "new-run"

    def test_monitor_usable_after_reset(self):
        monitor = QMLMonitor()
        monitor.update(step=0, loss=1.0)
        monitor.finish()
        monitor.reset()
        diagnosis = monitor.update(step=0, loss=1.0)
        assert diagnosis.degraded is False


class TestShouldStop:
    def test_false_before_any_update(self):
        monitor = QMLMonitor()
        assert monitor.should_stop() is False

    def test_false_with_default_warn_policy(self):
        monitor = QMLMonitor(policy="warn")
        monitor.update(step=0, loss=1.0)
        assert monitor.should_stop() is False

    def test_degraded_diagnosis_never_stops_unless_adaptive(self):
        monitor = QMLMonitor(policy="stop")
        monitor.state.latest_diagnosis = DiagnosisResult(
            issue=IssueType.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            severity="critical",
            evidence=[],
            recommendations=[],
            degraded=True,
            degraded_reason="synthetic test failure",
        )
        assert monitor.should_stop() is False

    def test_stop_policy_with_critical_severity_stops(self):
        monitor = QMLMonitor(policy="stop")
        monitor.state.latest_diagnosis = DiagnosisResult(
            issue=IssueType.POSSIBLE_BARREN_PLATEAU,
            confidence=0.9,
            severity="critical",
            evidence=["synthetic"],
            recommendations=["synthetic"],
        )
        assert monitor.should_stop() is True

    def test_stop_policy_with_info_severity_does_not_stop(self):
        monitor = QMLMonitor(policy="stop")
        monitor.update(step=0, loss=1.0)  # default diagnosis has severity "info"
        assert monitor.should_stop() is False


class TestContextManager:
    def test_enter_starts_and_exit_finishes(self):
        with QMLMonitor() as monitor:
            assert monitor.state.started is True
            monitor.update(step=0, loss=1.0)
        assert monitor.state.finished is True

    def test_enter_returns_the_same_monitor(self):
        monitor = QMLMonitor()
        with monitor as ctx:
            assert ctx is monitor

    def test_exit_does_not_suppress_exceptions(self):
        with pytest.raises(ValueError):
            with QMLMonitor() as monitor:
                monitor.update(step=0, loss=1.0)
                raise ValueError("boom")
        assert monitor.state.finished is True

    def test_reusable_in_a_second_with_block_after_reset(self):
        monitor = QMLMonitor()
        with monitor:
            monitor.update(step=0, loss=1.0)
        assert monitor.state.finished is True

        monitor.reset()

        with monitor:
            monitor.update(step=0, loss=1.0)
        assert monitor.state.finished is True
        assert monitor.state.step_count == 1

    def test_reentering_a_finished_monitor_without_reset_raises(self):
        monitor = QMLMonitor()
        with monitor:
            pass
        with pytest.raises(RuntimeError):
            with monitor:
                pass

    def test_exit_is_idempotent_if_already_finished_inside_block(self):
        with QMLMonitor() as monitor:
            monitor.update(step=0, loss=1.0)
            monitor.finish()  # explicit finish inside the block
        # __exit__ sees state.finished already True and does not error
        assert monitor.state.finished is True


class TestWatchDecorator:
    def test_watch_wraps_function_lifecycle(self):
        monitor = QMLMonitor()

        @monitor.watch
        def train():
            monitor.update(step=0, loss=1.0)
            return "done"

        result = train()
        assert result == "done"
        assert monitor.state.finished is True

    def test_watch_preserves_function_metadata(self):
        monitor = QMLMonitor()

        @monitor.watch
        def train():
            """Docstring."""

        assert train.__name__ == "train"
        assert train.__doc__ == "Docstring."

    def test_watch_passes_through_args_and_kwargs(self):
        monitor = QMLMonitor()

        @monitor.watch
        def train(a, b, *, c):
            return a + b + c

        assert train(1, 2, c=3) == 6

    def test_calling_watched_function_twice_without_reset_raises(self):
        # Each call to the decorated function is a full start()/finish()
        # cycle; calling it again re-enters __enter__ -> start(), which
        # raises because the monitor already finished. This mirrors plain
        # `with monitor:` reuse semantics (see TestContextManager).
        monitor = QMLMonitor()

        @monitor.watch
        def train():
            monitor.update(step=0, loss=1.0)

        train()
        with pytest.raises(RuntimeError):
            train()

    def test_watched_function_reusable_after_reset(self):
        monitor = QMLMonitor()

        @monitor.watch
        def train():
            monitor.update(step=0, loss=1.0)

        train()
        monitor.reset()
        train()  # no longer raises
        assert monitor.state.finished is True


class TestReporterHooks:
    def test_reporter_called_on_update_and_finish(self):
        events = []
        diagnoses = []
        finalized = []

        class FakeReporter:
            def record_event(self, event):
                events.append(event)

            def record_diagnosis(self, diagnosis):
                diagnoses.append(diagnosis)

            def finalize(self):
                finalized.append(True)

        monitor = QMLMonitor(reporter=FakeReporter())
        monitor.update(step=0, loss=1.0)
        monitor.finish()

        assert len(events) == 1
        assert len(diagnoses) == 1
        assert finalized == [True]

    def test_reporter_failure_does_not_propagate(self):
        class BrokenReporter:
            def record_event(self, event):
                raise RuntimeError("reporter is broken")

            def record_diagnosis(self, diagnosis):
                raise RuntimeError("reporter is broken")

            def finalize(self):
                raise RuntimeError("reporter is broken")

        monitor = QMLMonitor(reporter=BrokenReporter())
        diagnosis = monitor.update(step=0, loss=1.0)
        # record_event failure is caught by the same fail-open try/except
        # as any other processing error.
        assert diagnosis.degraded is True
        # finish()'s reporter failures are caught separately and don't raise.
        monitor.finish()
