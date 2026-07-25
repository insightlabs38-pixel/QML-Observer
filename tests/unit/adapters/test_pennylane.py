"""Unit tests for qml_observer.adapters.pennylane.adapter.PennyLaneAdapter.

Milestone 6, Issues #41-#45. Skipped entirely if the optional `pennylane`
dependency isn't installed (`pip install qml-observer[pennylane]`).
"""

import warnings

import pytest

pennylane = pytest.importorskip("pennylane")
qml = pennylane
from pennylane import numpy as pnp  # noqa: E402

from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter  # noqa: E402
from qml_observer.core.monitor import QMLMonitor  # noqa: E402
from qml_observer.schemas.circuit import CircuitMetadata  # noqa: E402
from qml_observer.schemas.diagnosis import DiagnosisResult  # noqa: E402

# PennyLane emits deprecation warnings for some legacy device kwargs across
# versions in the supported range; irrelevant noise for these tests.
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _analytic_circuit(diff_method="parameter-shift", n_wires=3):
    dev = qml.device("default.qubit", wires=n_wires, shots=None)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[1, 2])
        qml.RZ(params[2], wires=2)
        return qml.expval(qml.PauliZ(2))

    return circuit


def _shots_circuit(n_shots=500):
    dev = qml.device("default.qubit", wires=2, shots=n_shots)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    return circuit


class TestConstruction:
    def test_wraps_monitor(self):
        monitor = QMLMonitor()
        adapter = PennyLaneAdapter(monitor)
        assert adapter.monitor is monitor
        assert adapter.attached is False

    def test_rejects_non_monitor(self):
        with pytest.raises(TypeError):
            PennyLaneAdapter("not-a-monitor")

    def test_attaches_qnode_at_construction(self):
        circuit = _analytic_circuit()
        adapter = PennyLaneAdapter(QMLMonitor(), circuit)
        assert adapter.attached is True


class TestAttachDetach:
    def test_attach_returns_self(self):
        adapter = PennyLaneAdapter(QMLMonitor())
        circuit = _analytic_circuit()
        assert adapter.attach(circuit) is adapter
        assert adapter.attached is True

    def test_attach_rejects_non_qnode(self):
        adapter = PennyLaneAdapter(QMLMonitor())
        with pytest.raises(TypeError):
            adapter.attach(object())

    def test_detach_clears_qnode(self):
        adapter = PennyLaneAdapter(QMLMonitor(), _analytic_circuit())
        adapter.detach()
        assert adapter.attached is False


class TestRecordStep:
    def test_record_step_returns_diagnosis(self):
        adapter = PennyLaneAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0)
        assert isinstance(diagnosis, DiagnosisResult)

    def test_record_step_forwards_loss_and_gradients(self):
        monitor = QMLMonitor()
        adapter = PennyLaneAdapter(monitor)
        adapter.record_step(0, loss=0.5, gradients=pnp.array([0.1, 0.2]))

        obs = monitor.state.latest_observation
        assert obs.training_event.loss == 0.5
        assert obs.gradient is not None

    def test_record_step_without_attached_qnode_has_no_circuit_metadata(self):
        monitor = QMLMonitor()
        adapter = PennyLaneAdapter(monitor)
        adapter.record_step(0, loss=1.0, parameters=pnp.array([0.1, 0.2, 0.3]))
        assert monitor.state.latest_observation.circuit is None

    def test_record_multiple_steps_advances_monitor(self):
        monitor = QMLMonitor()
        adapter = PennyLaneAdapter(monitor, _analytic_circuit())
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)
        adapter.record_step(0, loss=1.0, parameters=params)
        adapter.record_step(1, loss=0.5, parameters=params)
        assert monitor.state.step_count == 2

    def test_fail_open_preserved_through_adapter(self):
        adapter = PennyLaneAdapter(QMLMonitor())
        diagnosis = adapter.record_step(0, loss=1.0, gradients=pnp.array([]))
        assert diagnosis.degraded is True


class TestParameterShiftMetadata:
    """Issue #42."""

    def test_gradient_method_recorded_as_parameter_shift(self):
        monitor = QMLMonitor()
        circuit = _analytic_circuit(diff_method="parameter-shift")
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

        adapter.record_step(0, loss=0.5, parameters=params)

        optimizer = monitor.state.latest_observation.optimizer
        assert optimizer is not None
        assert optimizer.gradient_method == "parameter-shift"

    def test_optimizer_name_and_learning_rate_forwarded(self):
        monitor = QMLMonitor()
        circuit = _analytic_circuit(diff_method="parameter-shift")
        adapter = PennyLaneAdapter(monitor, circuit, optimizer_name="Adam", learning_rate=0.05)
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

        adapter.record_step(0, loss=0.5, parameters=params)

        optimizer = monitor.state.latest_observation.optimizer
        assert optimizer.name == "Adam"
        assert optimizer.learning_rate == 0.05


class TestAdjointDifferentiation:
    """Issue #43."""

    def test_gradient_method_recorded_as_adjoint(self):
        monitor = QMLMonitor()
        circuit = _analytic_circuit(diff_method="adjoint")
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

        adapter.record_step(0, loss=0.5, parameters=params)

        optimizer = monitor.state.latest_observation.optimizer
        assert optimizer.gradient_method == "adjoint"

    def test_adjoint_circuit_still_extracts_circuit_metadata(self):
        monitor = QMLMonitor()
        circuit = _analytic_circuit(diff_method="adjoint")
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

        adapter.record_step(0, loss=0.5, parameters=params)

        circuit_meta = monitor.state.latest_observation.circuit
        assert circuit_meta is not None
        assert circuit_meta.n_qubits == 3


class TestFiniteShots:
    """Issue #44."""

    def test_shots_inferred_from_device_when_qnode_attached(self):
        monitor = QMLMonitor()
        circuit = _shots_circuit(n_shots=500)
        adapter = PennyLaneAdapter(monitor, circuit)
        x = pnp.array(0.3, requires_grad=True)

        adapter.record_step(0, loss=0.5, parameters=x)

        assert monitor.state.latest_observation.shots == 500

    def test_analytic_circuit_reports_no_shots(self):
        monitor = QMLMonitor()
        circuit = _analytic_circuit()
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

        adapter.record_step(0, loss=0.5, parameters=params)

        assert monitor.state.latest_observation.shots is None

    def test_explicit_shots_argument_overrides_inference(self):
        monitor = QMLMonitor()
        circuit = _shots_circuit(n_shots=500)
        adapter = PennyLaneAdapter(monitor, circuit)
        x = pnp.array(0.3, requires_grad=True)

        adapter.record_step(0, loss=0.5, parameters=x, shots=999)

        assert monitor.state.latest_observation.shots == 999

    def test_shots_inferred_without_parameters_from_device_default(self):
        monitor = QMLMonitor()
        circuit = _shots_circuit(n_shots=250)
        adapter = PennyLaneAdapter(monitor, circuit)

        adapter.record_step(0, loss=0.5)

        assert monitor.state.latest_observation.shots == 250


class TestExtractCircuitMetadata:
    """Issue #45."""

    def test_extracts_wires_gates_and_entangling_count(self):
        adapter = PennyLaneAdapter(QMLMonitor())
        circuit = _analytic_circuit(n_wires=3)
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)
        tape = qml.workflow.construct_tape(circuit)(params)

        meta = adapter.extract_circuit_metadata(tape)

        assert isinstance(meta, CircuitMetadata)
        assert meta.n_qubits == 3
        assert meta.n_gates == 5
        assert meta.n_entangling_gates == 2
        assert meta.n_parameters == 3

    def test_extracts_depth(self):
        adapter = PennyLaneAdapter(QMLMonitor())
        circuit = _analytic_circuit(n_wires=3)
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)
        tape = qml.workflow.construct_tape(circuit)(params)

        meta = adapter.extract_circuit_metadata(tape)

        assert meta.depth is not None
        assert meta.depth > 0

    def test_passes_through_ansatz_name_and_initialization(self):
        adapter = PennyLaneAdapter(QMLMonitor())
        circuit = _analytic_circuit(n_wires=3)
        params = pnp.array([0.1, 0.2, 0.3], requires_grad=True)
        tape = qml.workflow.construct_tape(circuit)(params)

        meta = adapter.extract_circuit_metadata(
            tape, ansatz_name="StronglyEntanglingLayers", initialization="random_uniform"
        )

        assert meta.ansatz_name == "StronglyEntanglingLayers"
        assert meta.initialization == "random_uniform"

    def test_handles_malformed_tape_gracefully(self):
        """A tape-like object missing most attributes should degrade to
        an (almost) empty CircuitMetadata rather than raising."""
        adapter = PennyLaneAdapter(QMLMonitor())

        class FakeTape:
            pass

        meta = adapter.extract_circuit_metadata(FakeTape())
        assert isinstance(meta, CircuitMetadata)
        assert meta.n_qubits is None
        assert meta.n_gates is None


class TestImportGuard:
    def test_clear_error_when_pennylane_missing(self, monkeypatch):
        import qml_observer.adapters.pennylane.adapter as adapter_module

        monkeypatch.setattr(adapter_module, "qml", None)
        monkeypatch.setattr(adapter_module, "_IMPORT_ERROR", ImportError("boom"))
        with pytest.raises(ImportError, match="pennylane"):
            adapter_module.PennyLaneAdapter(QMLMonitor())
