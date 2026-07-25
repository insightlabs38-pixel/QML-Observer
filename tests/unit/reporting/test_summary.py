"""Unit tests for qml_observer.reporting.summary (Milestone 7, Issue #49)."""

import numpy as np
import pytest

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.reporting.summary import build_run_summary
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType
from qml_observer.schemas.gradient import summarize_gradient
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent


def _diagnosis(issue=IssueType.HEALTHY, degraded=False):
    return DiagnosisResult(
        issue=issue,
        confidence=0.75,
        severity="info",
        evidence=["e1"],
        recommendations=["r1"],
        degraded=degraded,
        degraded_reason="boom" if degraded else None,
    )


class TestBuildRunSummary:
    def test_basic_fields(self):
        state = RunState(run_id="run-1", window_size=10)
        state.start_time = 0.0
        state.end_time = 10.0
        state.record(
            StepObservation(training_event=TrainingEvent(run_id="run-1", step=0, loss=1.0))
        )
        summary = build_run_summary(state, _diagnosis(), framework="pennylane")

        assert summary["run_id"] == "run-1"
        assert summary["framework"] == "pennylane"
        assert summary["duration"] == pytest.approx(10.0)
        assert summary["steps"] == 1
        assert summary["final_diagnosis"] == "healthy"
        assert summary["confidence"] == pytest.approx(0.75)
        assert summary["evidence"] == ["e1"]
        assert summary["recommendations"] == ["r1"]

    def test_none_fields_when_no_data(self):
        state = RunState(run_id="run-1", window_size=10)
        summary = build_run_summary(state, _diagnosis())
        assert summary["duration"] is None
        assert summary["circuit"] is None
        assert summary["optimizer"] is None
        assert summary["shots"] is None
        assert summary["gradient"] is None
        assert summary["loss_curve_summary"] is None
        assert summary["estimated_compute_saved"] is None

    def test_pulls_circuit_optimizer_shots_gradient_from_latest_observation(self):
        state = RunState(run_id="run-1", window_size=10)
        circuit = CircuitMetadata(n_qubits=4)
        optimizer = OptimizerMetadata(name="Adam", learning_rate=0.01)
        gradient = summarize_gradient(np.array([1.0, -1.0]))
        state.record(
            StepObservation(
                training_event=TrainingEvent(run_id="run-1", step=0, loss=1.0),
                gradient=gradient,
                circuit=circuit,
                optimizer=optimizer,
                shots=1024,
            )
        )
        summary = build_run_summary(state, _diagnosis())

        assert summary["circuit"]["n_qubits"] == 4
        assert summary["optimizer"]["name"] == "Adam"
        assert summary["shots"] == 1024
        assert summary["gradient"]["norm_l2"] == pytest.approx(gradient.norm_l2)

    def test_loss_curve_summary_over_window(self):
        state = RunState(run_id="run-1", window_size=10)
        for step, loss in enumerate([3.0, 2.0, 1.0]):
            state.record(
                StepObservation(
                    training_event=TrainingEvent(run_id="run-1", step=step, loss=loss)
                )
            )
        summary = build_run_summary(state, _diagnosis())
        curve = summary["loss_curve_summary"]
        assert curve["n_points"] == 3
        assert curve["first"] == 3.0
        assert curve["last"] == 1.0
        assert curve["min"] == 1.0
        assert curve["max"] == 3.0
        assert curve["has_non_finite"] is False

    def test_loss_curve_flags_non_finite(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(
            StepObservation(training_event=TrainingEvent(run_id="run-1", step=0, loss=float("nan")))
        )
        summary = build_run_summary(state, _diagnosis())
        assert summary["loss_curve_summary"]["has_non_finite"] is True

    def test_degraded_diagnosis_reflected(self):
        state = RunState(run_id="run-1", window_size=10)
        state.record(
            StepObservation(training_event=TrainingEvent(run_id="run-1", step=0, loss=1.0))
        )
        summary = build_run_summary(state, _diagnosis(degraded=True))
        assert summary["degraded"] is True
        assert summary["degraded_reason"] == "boom"

    def test_compute_saved_uses_planned_steps_and_mean_wall_time(self):
        state = RunState(run_id="run-1", window_size=10, planned_steps=10)
        for step in range(5):
            state.record(
                StepObservation(
                    training_event=TrainingEvent(
                        run_id="run-1", step=step, loss=1.0, wall_time=2.0
                    )
                )
            )
        summary = build_run_summary(state, _diagnosis())
        assert summary["estimated_compute_saved"] == pytest.approx((10 - 5) * 2.0)
