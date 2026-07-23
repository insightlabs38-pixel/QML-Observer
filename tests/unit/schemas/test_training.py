"""Unit tests for qml_observer.schemas.training.TrainingEvent."""

import pytest

from qml_observer.schemas.training import TrainingEvent


class TestConstruction:
    def test_minimal_construction(self):
        event = TrainingEvent(run_id="run-1", step=0)
        assert event.run_id == "run-1"
        assert event.step == 0
        assert event.loss is None
        assert event.epoch is None
        assert event.timestamp is None
        assert event.wall_time is None

    def test_full_construction(self):
        event = TrainingEvent(
            run_id="run-1",
            step=42,
            loss=0.123,
            epoch=3,
            timestamp=1721000000.0,
            wall_time=0.02,
        )
        assert event.step == 42
        assert event.loss == 0.123
        assert event.epoch == 3
        assert event.timestamp == 1721000000.0
        assert event.wall_time == 0.02


class TestLossToleratesNonFiniteValues:
    """Per addendum §7: NaN/Inf loss is a meaningful diverging-optimizer
    signal for the detector layer, not a schema-level error."""

    def test_nan_loss_is_allowed(self):
        event = TrainingEvent(run_id="run-1", step=0, loss=float("nan"))
        assert event.loss != event.loss  # NaN != NaN

    def test_inf_loss_is_allowed(self):
        event = TrainingEvent(run_id="run-1", step=0, loss=float("inf"))
        assert event.loss == float("inf")

    def test_negative_inf_loss_is_allowed(self):
        event = TrainingEvent(run_id="run-1", step=0, loss=float("-inf"))
        assert event.loss == float("-inf")


class TestValidation:
    def test_empty_run_id_raises(self):
        with pytest.raises(ValueError, match="run_id"):
            TrainingEvent(run_id="", step=0)

    def test_non_str_run_id_raises(self):
        with pytest.raises(TypeError):
            TrainingEvent(run_id=123, step=0)  # type: ignore[arg-type]

    def test_negative_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            TrainingEvent(run_id="run-1", step=-1)

    def test_non_int_step_raises(self):
        with pytest.raises(TypeError):
            TrainingEvent(run_id="run-1", step=1.5)  # type: ignore[arg-type]

    def test_bool_step_raises(self):
        """bool is a subclass of int in Python; must be rejected explicitly."""
        with pytest.raises(ValueError):
            TrainingEvent(run_id="run-1", step=True)  # type: ignore[arg-type]

    def test_non_numeric_loss_raises(self):
        with pytest.raises(TypeError):
            TrainingEvent(run_id="run-1", step=0, loss="bad")  # type: ignore[arg-type]

    def test_negative_epoch_raises(self):
        with pytest.raises(ValueError, match="epoch"):
            TrainingEvent(run_id="run-1", step=0, epoch=-1)

    def test_negative_wall_time_raises(self):
        with pytest.raises(ValueError, match="wall_time"):
            TrainingEvent(run_id="run-1", step=0, wall_time=-0.5)

    def test_nan_wall_time_is_tolerated(self):
        """Instrumentation failure producing NaN wall_time shouldn't hard-fail."""
        event = TrainingEvent(run_id="run-1", step=0, wall_time=float("nan"))
        assert event.wall_time != event.wall_time

    def test_non_numeric_timestamp_raises(self):
        with pytest.raises(TypeError):
            TrainingEvent(run_id="run-1", step=0, timestamp="not-a-number")  # type: ignore[arg-type]
