"""Unit tests for qml_observer.adapters.generic.GenericAdapter."""

import numpy as np
import pytest

from qml_observer.adapters.generic import GenericAdapter
from qml_observer.core.monitor import QMLMonitor
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult
from qml_observer.schemas.optimizer import OptimizerMetadata


class TestConstruction:
    def test_wraps_monitor(self):
        monitor = QMLMonitor()
        adapter = GenericAdapter(monitor)
        assert adapter.monitor is monitor

    def test_rejects_non_monitor(self):
        with pytest.raises(TypeError):
            GenericAdapter("not-a-monitor")


class TestRecord:
    def test_record_returns_diagnosis(self):
        adapter = GenericAdapter(QMLMonitor())
        diagnosis = adapter.record(0, loss=1.0)
        assert isinstance(diagnosis, DiagnosisResult)

    def test_record_forwards_to_monitor_state(self):
        monitor = QMLMonitor()
        adapter = GenericAdapter(monitor)
        adapter.record(0, loss=0.5, gradients=np.array([0.1, 0.2]))

        obs = monitor.state.latest_observation
        assert obs.training_event.step == 0
        assert obs.training_event.loss == 0.5
        assert obs.gradient is not None

    def test_record_forwards_all_optional_metadata(self):
        monitor = QMLMonitor()
        adapter = GenericAdapter(monitor)
        circuit = CircuitMetadata(n_qubits=4)
        optimizer = OptimizerMetadata(name="Adam")

        adapter.record(
            0,
            loss=0.5,
            gradients=np.array([0.1]),
            parameters=[0.9],
            circuit=circuit,
            optimizer=optimizer,
            shots=1024,
        )

        obs = monitor.state.latest_observation
        assert obs.circuit is circuit
        assert obs.optimizer is optimizer
        assert obs.shots == 1024
        assert obs.parameters == [0.9]

    def test_record_multiple_steps_advances_monitor(self):
        monitor = QMLMonitor()
        adapter = GenericAdapter(monitor)
        adapter.record(0, loss=1.0)
        adapter.record(1, loss=0.5)
        assert monitor.state.step_count == 2

    def test_fail_open_preserved_through_adapter(self):
        adapter = GenericAdapter(QMLMonitor())
        diagnosis = adapter.record(0, loss=1.0, gradients=np.array([]))
        assert diagnosis.degraded is True

    def test_record_after_finish_raises(self):
        monitor = QMLMonitor()
        adapter = GenericAdapter(monitor)
        adapter.record(0, loss=1.0)
        monitor.finish()
        with pytest.raises(RuntimeError):
            adapter.record(1, loss=0.5)


class TestIntegrationWithContextManager:
    def test_adapter_inside_monitor_context_manager(self):
        with QMLMonitor() as monitor:
            adapter = GenericAdapter(monitor)
            for step in range(3):
                adapter.record(step, loss=1.0 / (step + 1))
        assert monitor.state.finished is True
        assert monitor.state.step_count == 3
