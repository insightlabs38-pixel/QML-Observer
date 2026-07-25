"""Unit tests for qml_observer.reporting.export (Milestone 7, Issue #51)."""

import json

import pytest

from qml_observer.core.state import RunState
from qml_observer.reporting.export import (
    estimate_compute_saved,
    estimate_compute_saved_from_state,
    export_summary_json,
    format_compute_saved,
)


class TestEstimateComputeSaved:
    def test_matches_addendum_formula(self):
        # saved = (planned - actual) * mean_wall_time_per_step
        assert estimate_compute_saved(1000, 200, 2.0) == pytest.approx(1600.0)

    def test_none_when_planned_steps_missing(self):
        assert estimate_compute_saved(None, 200, 2.0) is None

    def test_none_when_mean_wall_time_missing(self):
        assert estimate_compute_saved(1000, 200, None) is None

    def test_clamped_to_zero_when_run_completed(self):
        assert estimate_compute_saved(100, 100, 2.0) == 0.0

    def test_clamped_to_zero_when_run_exceeded_plan(self):
        assert estimate_compute_saved(100, 150, 2.0) == 0.0

    def test_rejects_negative_actual_steps(self):
        with pytest.raises(ValueError):
            estimate_compute_saved(100, -1, 2.0)

    def test_zero_actual_steps_is_valid(self):
        assert estimate_compute_saved(100, 0, 1.0) == pytest.approx(100.0)


class TestEstimateComputeSavedFromState:
    def test_uses_state_planned_steps_and_mean_wall_time(self):
        state = RunState(run_id="run-1", window_size=10, planned_steps=100)
        from qml_observer.core.events import StepObservation
        from qml_observer.schemas.training import TrainingEvent

        for step in range(5):
            state.record(
                StepObservation(
                    training_event=TrainingEvent(
                        run_id="run-1", step=step, loss=1.0, wall_time=2.0
                    )
                )
            )

        saved = estimate_compute_saved_from_state(state)
        assert saved == pytest.approx((100 - 5) * 2.0)

    def test_none_without_planned_steps(self):
        state = RunState(run_id="run-1", window_size=10)
        assert estimate_compute_saved_from_state(state) is None

    def test_actual_steps_override(self):
        state = RunState(run_id="run-1", window_size=10, planned_steps=100)
        from qml_observer.core.events import StepObservation
        from qml_observer.schemas.training import TrainingEvent

        state.record(
            StepObservation(
                training_event=TrainingEvent(run_id="run-1", step=0, wall_time=2.0)
            )
        )
        saved = estimate_compute_saved_from_state(state, actual_steps_at_stop=10)
        assert saved == pytest.approx((100 - 10) * 2.0)


class TestFormatComputeSaved:
    def test_none_is_unknown(self):
        assert "unknown" in format_compute_saved(None)

    def test_seconds(self):
        assert format_compute_saved(30) == "~30 seconds"

    def test_minutes(self):
        assert format_compute_saved(150) == "~2.5 minutes"

    def test_hours(self):
        assert format_compute_saved(7200) == "~2.0 hours"

    def test_sub_second_rounds_to_zero(self):
        assert format_compute_saved(0.2) == "~0 seconds"


class TestExportSummaryJson:
    def test_writes_formatted_json(self, tmp_path):
        summary = {"run_id": "run-1", "steps": 10}
        path = export_summary_json(summary, tmp_path / "summary.json")
        assert path.exists()
        assert json.loads(path.read_text()) == summary

    def test_creates_parent_directories(self, tmp_path):
        path = export_summary_json({"a": 1}, tmp_path / "nested" / "summary.json")
        assert path.exists()
