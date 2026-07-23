"""Unit tests for qml_observer.core.events.StepObservation."""

import pytest

from qml_observer.core.events import StepObservation
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.gradient import summarize_gradient
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent


def _event(step=0):
    return TrainingEvent(run_id="run-1", step=step)


class TestConstruction:
    def test_minimal(self):
        obs = StepObservation(training_event=_event())
        assert obs.gradient is None
        assert obs.circuit is None
        assert obs.optimizer is None
        assert obs.shots is None
        assert obs.parameters is None

    def test_full(self):
        grad = summarize_gradient([0.1, 0.2, 0.3])
        circuit = CircuitMetadata(n_qubits=4)
        optimizer = OptimizerMetadata(name="Adam")
        obs = StepObservation(
            training_event=_event(),
            gradient=grad,
            circuit=circuit,
            optimizer=optimizer,
            shots=1024,
            parameters=[0.1, 0.2],
        )
        assert obs.gradient is grad
        assert obs.circuit is circuit
        assert obs.optimizer is optimizer
        assert obs.shots == 1024
        assert obs.parameters == [0.1, 0.2]


class TestValidation:
    def test_wrong_training_event_type(self):
        with pytest.raises(TypeError):
            StepObservation(training_event="not-an-event")

    def test_wrong_gradient_type(self):
        with pytest.raises(TypeError):
            StepObservation(training_event=_event(), gradient="nope")

    def test_wrong_circuit_type(self):
        with pytest.raises(TypeError):
            StepObservation(training_event=_event(), circuit="nope")

    def test_wrong_optimizer_type(self):
        with pytest.raises(TypeError):
            StepObservation(training_event=_event(), optimizer="nope")

    def test_negative_shots(self):
        with pytest.raises(ValueError):
            StepObservation(training_event=_event(), shots=-1)

    def test_non_int_shots(self):
        with pytest.raises(TypeError):
            StepObservation(training_event=_event(), shots=1.5)

    def test_bool_shots_rejected(self):
        with pytest.raises(TypeError):
            StepObservation(training_event=_event(), shots=True)
