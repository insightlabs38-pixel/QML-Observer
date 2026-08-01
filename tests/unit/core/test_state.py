"""Unit tests for qml_observer.core.state.RunState."""

import pytest

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.schemas.training import TrainingEvent


def _obs(step, wall_time=None):
    return StepObservation(
        training_event=TrainingEvent(run_id="run-1", step=step, wall_time=wall_time)
    )


class TestConstruction:
    def test_defaults(self):
        state = RunState(run_id="run-1", window_size=10)
        assert state.run_id == "run-1"
        assert state.step_count == 0
        assert state.window == []
        assert state.latest_observation is None
        assert state.started is False
        assert state.finished is False

    def test_invalid_window_size(self):
        with pytest.raises(ValueError):
            RunState(run_id="run-1", window_size=0)

    def test_non_int_window_size(self):
        with pytest.raises(TypeError):
            RunState(run_id="run-1", window_size=1.5)


class TestRecord:
    def test_record_increments_step_count(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(_obs(0))
        state.record(_obs(1))
        assert state.step_count == 2
        assert len(state.window) == 2

    def test_window_bounded_by_window_size(self):
        state = RunState(run_id="run-1", window_size=3)
        for i in range(10):
            state.record(_obs(i))
        assert state.step_count == 10  # total steps, unbounded
        assert len(state.window) == 3  # windowed view, bounded
        assert [o.training_event.step for o in state.window] == [7, 8, 9]

    def test_latest_observation(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(_obs(0))
        state.record(_obs(1))
        assert state.latest_observation.training_event.step == 1

    def test_record_rejects_wrong_type(self):
        state = RunState(run_id="run-1", window_size=10)
        with pytest.raises(TypeError):
            state.record("not-an-observation")


class TestMeanWallTime:
    def test_no_wall_times_returns_none(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(_obs(0))
        assert state.mean_wall_time() is None

    def test_computes_mean_ignoring_none(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(_obs(0, wall_time=1.0))
        state.record(_obs(1, wall_time=None))
        state.record(_obs(2, wall_time=3.0))
        assert state.mean_wall_time() == pytest.approx(2.0)


class TestReset:
    def test_reset_clears_everything(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(_obs(0))
        state.started = True
        state.finished = True
        state.start_time = 1.0
        state.end_time = 2.0

        state.reset()

        assert state.step_count == 0
        assert state.window == []
        assert state.started is False
        assert state.finished is False
        assert state.start_time is None
        assert state.end_time is None
        assert state.latest_diagnosis is None
        # run_id/window_size/planned_steps are untouched by reset()
        assert state.run_id == "run-1"


class TestSeedStepCount:
    """Milestone 13, Issue #97 ("Automatic resume")."""

    def test_seeds_step_count_on_fresh_state(self):
        state = RunState(run_id="run-1", window_size=10)
        state.seed_step_count(42)
        assert state.step_count == 42

    def test_seeds_zero(self):
        state = RunState(run_id="run-1", window_size=10)
        state.seed_step_count(0)
        assert state.step_count == 0

    def test_negative_step_raises(self):
        state = RunState(run_id="run-1", window_size=10)
        with pytest.raises(ValueError, match="step"):
            state.seed_step_count(-1)

    def test_raises_if_observations_already_recorded(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(_obs(0))
        with pytest.raises(ValueError, match="no recorded observations"):
            state.seed_step_count(10)

    def test_allowed_again_after_reset(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(_obs(0))
        state.reset()
        state.seed_step_count(5)  # does not raise
        assert state.step_count == 5

    def test_seeding_does_not_touch_window(self):
        state = RunState(run_id="run-1", window_size=10)
        state.seed_step_count(50)
        assert state.window == []
        assert state.latest_observation is None

    def test_record_after_seeding_continues_from_seeded_count(self):
        state = RunState(run_id="run-1", window_size=10)
        state.seed_step_count(50)
        state.record(_obs(50))
        assert state.step_count == 51
