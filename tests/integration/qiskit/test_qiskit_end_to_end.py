"""Integration tests for the Qiskit adapter (Milestone 8, Issue #62).

Unlike `tests/unit/adapters/test_qiskit.py`, which exercises `QiskitAdapter`
in isolation with hand-built circuits/gradients, these tests drive real
Qiskit training: a real `QuantumCircuit` ansatz, a real `Estimator`
primitive, real parameter-shift-computed gradients, the real detector
stack, and the real `ActionPolicy` -- end to end, exactly as
`examples/qiskit/` does. A separate suite covers the callback-integration
path against a real `qiskit-machine-learning` `VQC` trainer. Skipped
entirely if the optional `qiskit`/`qiskit-machine-learning` dependencies
aren't installed.
"""

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit.circuit import Parameter, QuantumCircuit  # noqa: E402
from qiskit.circuit.library import efficient_su2  # noqa: E402
from qiskit.primitives import StatevectorEstimator  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from qml_observer import QMLMonitor  # noqa: E402
from qml_observer.adapters.qiskit.adapter import QiskitAdapter  # noqa: E402
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector  # noqa: E402
from qml_observer.detectors.convergence import ConvergenceDetector  # noqa: E402
from qml_observer.detectors.stagnation import StagnationDetector  # noqa: E402
from qml_observer.schemas.diagnosis import IssueType  # noqa: E402

PATIENCE = 15
_estimator = StatevectorEstimator()


def _detectors():
    return [
        BarrenPlateauDetector(patience=PATIENCE),
        StagnationDetector(patience=PATIENCE),
        ConvergenceDetector(patience=PATIENCE, loss_threshold=1e-2),
    ]


def _healthy_ansatz():
    return efficient_su2(2, reps=1)


def _plateau_ansatz():
    """RZ-only ansatz on |00>, Z-basis measurement: gradient is
    (numerically) exactly zero regardless of parameters -- same
    engineered-plateau construction as `examples/qiskit/barren_plateau_demo.py`
    and the PennyLane integration tests, without needing 15-20+ qubits to
    reproduce a real barren plateau."""
    circuit = QuantumCircuit(2)
    p0, p1 = Parameter("p0"), Parameter("p1")
    circuit.rz(p0, 0)
    circuit.rz(p1, 1)
    return circuit


def _observable(n_qubits):
    return SparsePauliOp("Z" + "I" * (n_qubits - 1))


def _energy(ansatz, observable, params):
    result = _estimator.run([(ansatz, observable, [params])]).result()
    return float(result[0].data.evs[0])


def _param_shift_gradient(ansatz, observable, params):
    grad = np.zeros_like(params)
    shift = np.pi / 2
    for i in range(len(params)):
        plus, minus = params.copy(), params.copy()
        plus[i] += shift
        minus[i] -= shift
        grad[i] = 0.5 * (_energy(ansatz, observable, plus) - _energy(ansatz, observable, minus))
    return grad


def _train(monitor, adapter, ansatz, params, *, max_steps, learning_rate=0.4):
    """Shared training-loop driver: real Estimator + real parameter-shift gradients."""
    observable = _observable(ansatz.num_qubits)
    diagnosis = None
    for step in range(max_steps):
        loss = _energy(ansatz, observable, params)
        gradients = _param_shift_gradient(ansatz, observable, params)
        diagnosis = adapter.record_step(step, loss, gradients, parameters=params)
        params = params - learning_rate * gradients
        if monitor.should_stop():
            break
    return diagnosis, params


class TestHealthyConvergenceEndToEnd:
    def test_healthy_run_is_not_stopped_early(self):
        monitor = QMLMonitor(
            detectors=_detectors(), policy="stop", window_size=50, planned_steps=1000
        )
        ansatz = _healthy_ansatz()
        adapter = QiskitAdapter(monitor, ansatz, optimizer_name="GradientDescent")
        params = np.array([0.9, -0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        diagnosis, _ = _train(monitor, adapter, ansatz, params, max_steps=100)

        assert monitor.should_stop() is False
        assert diagnosis.issue == IssueType.CONVERGED

    def test_healthy_run_reaches_low_loss(self):
        monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
        ansatz = _healthy_ansatz()
        adapter = QiskitAdapter(monitor, ansatz)
        params = np.array([0.9, -0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        observable = _observable(ansatz.num_qubits)
        _, final_params = _train(monitor, adapter, ansatz, params, max_steps=100)

        assert _energy(ansatz, observable, final_params) == pytest.approx(-1.0, abs=1e-2)


class TestEngineeredPlateauEndToEnd:
    def test_plateau_run_is_stopped_early(self):
        monitor = QMLMonitor(
            detectors=_detectors(), policy="stop", window_size=50, planned_steps=1000
        )
        ansatz = _plateau_ansatz()
        adapter = QiskitAdapter(monitor, ansatz, optimizer_name="GradientDescent")
        params = np.array([0.3, 0.5])

        diagnosis, _ = _train(monitor, adapter, ansatz, params, max_steps=200)

        assert monitor.should_stop() is True
        assert diagnosis.issue == IssueType.POSSIBLE_BARREN_PLATEAU
        assert monitor.state.step_count < 200

    def test_plateau_run_stop_action_actually_fires(self):
        """Not just should_stop()==True: a real StopAction ran via ActionPolicy."""
        monitor = QMLMonitor(
            detectors=_detectors(), policy="stop", window_size=50, planned_steps=1000
        )
        ansatz = _plateau_ansatz()
        adapter = QiskitAdapter(monitor, ansatz)
        params = np.array([0.3, 0.5])

        _train(monitor, adapter, ansatz, params, max_steps=200)

        result = monitor.latest_action_result()
        assert result is not None
        assert result.action_name == "stop"
        assert result.executed is True


class TestCircuitAndOptimizerMetadataEndToEnd:
    def test_circuit_metadata_recorded_through_full_loop(self):
        monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
        ansatz = _healthy_ansatz()
        adapter = QiskitAdapter(monitor, ansatz)
        params = np.array([0.9, -0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        _train(monitor, adapter, ansatz, params, max_steps=10)

        circuit_meta = monitor.state.latest_observation.circuit
        assert circuit_meta.n_qubits == 2
        assert circuit_meta.n_parameters == ansatz.num_parameters

    def test_optimizer_metadata_recorded_through_full_loop(self):
        monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
        ansatz = _healthy_ansatz()
        adapter = QiskitAdapter(
            monitor, ansatz, optimizer_name="GradientDescent", learning_rate=0.4
        )
        params = np.array([0.9, -0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        _train(monitor, adapter, ansatz, params, max_steps=10)

        optimizer = monitor.state.latest_observation.optimizer
        assert optimizer.name == "GradientDescent"
        assert optimizer.learning_rate == 0.4


class TestFailOpenWithRealCircuit:
    def test_bad_gradient_shape_degrades_without_crashing_training(self):
        monitor = QMLMonitor(detectors=_detectors(), policy="warn", window_size=50)
        ansatz = _healthy_ansatz()
        adapter = QiskitAdapter(monitor, ansatz)
        params = np.array([0.9, -0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        observable = _observable(ansatz.num_qubits)

        loss0 = _energy(ansatz, observable, params)
        adapter.record_step(
            0, loss0, _param_shift_gradient(ansatz, observable, params), parameters=params
        )

        diagnosis = adapter.record_step(
            1, _energy(ansatz, observable, params), np.array([]), parameters=params
        )
        assert diagnosis.degraded is True

        loss2 = _energy(ansatz, observable, params)
        diagnosis2 = adapter.record_step(
            2, loss2, _param_shift_gradient(ansatz, observable, params), parameters=params
        )
        assert diagnosis2.degraded is False
        assert monitor.state.step_count == 2


class TestVQCCallbackIntegrationEndToEnd:
    """Issue #59, exercised against a real qiskit-machine-learning trainer."""

    def test_vqc_fit_drives_monitor_via_callback(self):
        qiskit_machine_learning = pytest.importorskip("qiskit_machine_learning")
        from qiskit.circuit.library import zz_feature_map
        from qiskit_machine_learning.algorithms.classifiers import VQC
        from qiskit_machine_learning.optimizers import COBYLA

        del qiskit_machine_learning  # only needed for the importorskip check

        monitor = QMLMonitor(policy="log", window_size=50)
        optimizer = COBYLA(maxiter=15)
        adapter = QiskitAdapter(monitor, optimizer=optimizer)

        feature_map = zz_feature_map(2)
        ansatz = efficient_su2(2, reps=1)
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=adapter.callback,
        )
        adapter.attach(vqc)

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(16, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        vqc.fit(X, y)

        assert monitor.state.step_count == 15
        obs = monitor.state.latest_observation
        assert obs.training_event.loss is not None
        assert obs.circuit is not None
        assert obs.optimizer.name == "COBYLA"
        assert obs.optimizer.gradient_method == "gradient-free"

    def test_vqc_callback_survives_finish(self):
        qiskit_machine_learning = pytest.importorskip("qiskit_machine_learning")
        from qiskit.circuit.library import zz_feature_map
        from qiskit_machine_learning.algorithms.classifiers import VQC
        from qiskit_machine_learning.optimizers import COBYLA

        del qiskit_machine_learning

        monitor = QMLMonitor(policy="log", window_size=50)
        optimizer = COBYLA(maxiter=10)
        adapter = QiskitAdapter(monitor, optimizer=optimizer)

        feature_map = zz_feature_map(2)
        ansatz = efficient_su2(2, reps=1)
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            callback=adapter.callback,
        )
        adapter.attach(vqc)

        rng = np.random.default_rng(1)
        X = rng.uniform(-1, 1, size=(16, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        vqc.fit(X, y)

        final = monitor.finish()
        assert final.degraded is False
