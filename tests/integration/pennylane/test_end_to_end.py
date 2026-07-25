"""Integration tests for the PennyLane adapter (Milestone 6, Issue #47).

Unlike `tests/unit/adapters/test_pennylane.py`, which exercises
`PennyLaneAdapter` in isolation with hand-built tapes/gradients, these
tests drive a full, real training loop: a real `QNode`, a real PennyLane
optimizer, real `qml.grad()`-computed gradients, the real detector stack,
and the real `ActionPolicy` -- end to end, exactly as `examples/pennylane/`
does. Skipped entirely if the optional `pennylane` dependency isn't
installed.
"""

import warnings

import pytest

pennylane = pytest.importorskip("pennylane")
qml = pennylane
from pennylane import numpy as pnp  # noqa: E402

from qml_observer import QMLMonitor  # noqa: E402
from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter  # noqa: E402
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector  # noqa: E402
from qml_observer.detectors.convergence import ConvergenceDetector  # noqa: E402
from qml_observer.detectors.stagnation import StagnationDetector  # noqa: E402
from qml_observer.schemas.diagnosis import IssueType  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)

PATIENCE = 15


def _detectors():
    return [
        BarrenPlateauDetector(patience=PATIENCE),
        StagnationDetector(patience=PATIENCE),
        ConvergenceDetector(patience=PATIENCE, loss_threshold=1e-2),
    ]


def _healthy_circuit(shots=None):
    dev = qml.device("default.qubit", wires=2, shots=shots)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    return circuit


def _plateau_circuit():
    """RZ-only ansatz on |0>: gradient is (numerically) exactly zero
    regardless of parameters -- stands in for a real barren plateau
    without needing 15-20+ qubits to reproduce (see
    examples/pennylane/barren_plateau_demo.py for the full rationale)."""
    dev = qml.device("default.qubit", wires=2, shots=None)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        qml.RZ(params[0], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.expval(qml.PauliZ(0))

    return circuit


def _train(monitor, adapter, circuit, params, *, max_steps, stepsize=0.4):
    """Shared training-loop driver: real qml.grad + real GradientDescentOptimizer."""
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    diagnosis = None
    for step in range(max_steps):
        gradients = qml.grad(circuit)(params)
        loss = circuit(params)
        diagnosis = adapter.record_step(step, float(loss), gradients, parameters=params)
        params = opt.step(circuit, params)
        if monitor.should_stop():
            break
    return diagnosis, params


class TestHealthyConvergenceEndToEnd:
    def test_healthy_run_is_not_stopped_early(self):
        monitor = QMLMonitor(
            detectors=_detectors(), policy="stop", window_size=50, planned_steps=1000
        )
        circuit = _healthy_circuit()
        adapter = PennyLaneAdapter(monitor, circuit, optimizer_name="GradientDescent")
        params = pnp.array([0.9, -0.6], requires_grad=True)

        diagnosis, _ = _train(monitor, adapter, circuit, params, max_steps=100)

        assert monitor.should_stop() is False
        assert diagnosis.issue == IssueType.CONVERGED

    def test_healthy_run_reaches_low_loss(self):
        monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
        circuit = _healthy_circuit()
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.9, -0.6], requires_grad=True)

        _, final_params = _train(monitor, adapter, circuit, params, max_steps=100)

        assert float(circuit(final_params)) == pytest.approx(-1.0, abs=1e-2)


class TestEngineeredPlateauEndToEnd:
    def test_plateau_run_is_stopped_early(self):
        monitor = QMLMonitor(
            detectors=_detectors(), policy="stop", window_size=50, planned_steps=1000
        )
        circuit = _plateau_circuit()
        adapter = PennyLaneAdapter(monitor, circuit, optimizer_name="GradientDescent")
        params = pnp.array([0.3, 0.5], requires_grad=True)

        diagnosis, _ = _train(monitor, adapter, circuit, params, max_steps=200)

        assert monitor.should_stop() is True
        assert diagnosis.issue == IssueType.POSSIBLE_BARREN_PLATEAU
        # Stopped comfortably before the full planned budget.
        assert monitor.state.step_count < 200

    def test_plateau_run_stop_action_actually_fires(self):
        """Not just should_stop()==True: a real StopAction ran via ActionPolicy."""
        monitor = QMLMonitor(
            detectors=_detectors(), policy="stop", window_size=50, planned_steps=1000
        )
        circuit = _plateau_circuit()
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.3, 0.5], requires_grad=True)

        _train(monitor, adapter, circuit, params, max_steps=200)

        result = monitor.latest_action_result()
        assert result is not None
        assert result.action_name == "stop"
        assert result.executed is True


class TestParameterShiftAndAdjointEndToEnd:
    def test_parameter_shift_end_to_end(self):
        monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
        circuit = _healthy_circuit()
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.9, -0.6], requires_grad=True)

        _train(monitor, adapter, circuit, params, max_steps=30)

        optimizer = monitor.state.latest_observation.optimizer
        assert optimizer.gradient_method == "parameter-shift"

    def test_adjoint_end_to_end(self):
        dev = qml.device("default.qubit", wires=2, shots=None)

        @qml.qnode(dev, diff_method="adjoint")
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(params[1], wires=1)
            return qml.expval(qml.PauliZ(1))

        monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.9, -0.6], requires_grad=True)

        diagnosis, final_params = _train(monitor, adapter, circuit, params, max_steps=100)

        optimizer = monitor.state.latest_observation.optimizer
        assert optimizer.gradient_method == "adjoint"
        assert float(circuit(final_params)) == pytest.approx(-1.0, abs=1e-2)
        assert diagnosis.issue == IssueType.CONVERGED


class TestFiniteShotsEndToEnd:
    def test_finite_shots_recorded_through_full_loop(self):
        monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
        circuit = _healthy_circuit(shots=100)
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.9, -0.6], requires_grad=True)

        _train(monitor, adapter, circuit, params, max_steps=20)

        assert monitor.state.latest_observation.shots == 100

    def test_finite_shots_does_not_crash_diagnosis_pipeline(self):
        """Noisy finite-shot gradients must flow through detectors/diagnosis
        without ever degrading (unlike a genuine data error, shot noise is
        expected, valid data)."""
        monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
        circuit = _healthy_circuit(shots=20)
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.9, -0.6], requires_grad=True)

        diagnosis, _ = _train(monitor, adapter, circuit, params, max_steps=40)

        assert diagnosis.degraded is False


class TestFailOpenWithRealQNode:
    def test_bad_gradient_shape_degrades_without_crashing_training(self):
        monitor = QMLMonitor(detectors=_detectors(), policy="warn", window_size=50)
        circuit = _healthy_circuit()
        adapter = PennyLaneAdapter(monitor, circuit)
        params = pnp.array([0.9, -0.6], requires_grad=True)

        # A legitimate step, then a step with a broken (empty) gradient array,
        # then another legitimate step -- the loop must survive all three.
        loss0 = circuit(params)
        adapter.record_step(0, float(loss0), qml.grad(circuit)(params), parameters=params)

        diagnosis = adapter.record_step(1, float(circuit(params)), pnp.array([]), parameters=params)
        assert diagnosis.degraded is True

        loss2 = circuit(params)
        diagnosis2 = adapter.record_step(
            2, float(loss2), qml.grad(circuit)(params), parameters=params
        )
        assert diagnosis2.degraded is False
        # The failed step (step=1) never made it into the recorded window --
        # summarize_gradient() raises before an observation is constructed --
        # so only the two legitimate steps (0 and 2) are counted. What matters
        # for fail-open is that step 2 was reached and processed at all.
        assert monitor.state.step_count == 2
