"""Milestone 8, Issue #61: the Qiskit version of the "critical MVP demo"
(blueprint Volume XX), matching `examples/pennylane/barren_plateau_demo.py`.

Runs two training scenarios back to back with real detectors wired in and
`policy="stop"`:

  1. A healthy, well-conditioned circuit that converges normally -- it
     must run to completion and must NOT be stopped early.
  2. An engineered plateau-like circuit -- deliberately built from only
     RZ rotations measured in the Z basis, so its expectation value (and
     therefore its gradient) is invariant to every parameter. Same
     rationale as the PennyLane demo: this reproduces the "collapsed
     gradient + stagnant loss" signature a real barren plateau produces,
     without needing the ~15-20+ qubits an actual random-circuit barren
     plateau needs to reproduce reliably.

Loss and gradients are computed manually via Qiskit's `Estimator`
primitive and the parameter-shift rule (see `basic_monitor.py`) -- the
adapter only observes the results, per the blueprint's core rule.

The second run should be caught and stopped well before it would
otherwise reach its planned step budget, and the script reports the
estimated compute saved using the addendum's resolved formula:

    saved = (planned_total_steps - actual_steps_at_stop) * mean_wall_time_per_step

Run with:
    python examples/qiskit/barren_plateau_demo.py
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qml_observer import QMLMonitor
from qml_observer.adapters.qiskit.adapter import QiskitAdapter
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.convergence import ConvergenceDetector
from qml_observer.detectors.stagnation import StagnationDetector

PATIENCE = 15
PLANNED_STEPS = 5_000  # the compute budget this run would otherwise be given
MAX_DEMO_STEPS = 200  # hard cap so the demo itself can't run away
LEARNING_RATE = 0.4

_estimator = StatevectorEstimator()


def _detectors() -> list:
    """A fresh detector set per run -- detectors are stateful, so each
    `QMLMonitor` needs its own instances rather than a shared list."""
    return [
        BarrenPlateauDetector(patience=PATIENCE),
        StagnationDetector(patience=PATIENCE),
        ConvergenceDetector(patience=PATIENCE, loss_threshold=1e-2),
    ]


def healthy_ansatz():
    """A simple, well-conditioned 2-qubit ansatz that converges normally."""
    return efficient_su2(2, reps=1)


def plateau_ansatz():
    """An engineered plateau ansatz: RZ-only, applied to |00>. `RZ` is
    diagonal, so it leaves |00> globally unchanged up to phase -- the
    Z-basis expectation value (and therefore the gradient) is invariant to
    every parameter, exactly the failure signature a real barren plateau
    produces."""
    circuit = QuantumCircuit(2)
    p0, p1 = Parameter("p0"), Parameter("p1")
    circuit.rz(p0, 0)
    circuit.rz(p1, 1)
    return circuit


def _observable(n_qubits: int) -> SparsePauliOp:
    return SparsePauliOp("Z" + "I" * (n_qubits - 1))


def energy(ansatz, observable: SparsePauliOp, params: np.ndarray) -> float:
    result = _estimator.run([(ansatz, observable, [params])]).result()
    return float(result[0].data.evs[0])


def parameter_shift_gradient(ansatz, observable: SparsePauliOp, params: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(params)
    shift = np.pi / 2
    for i in range(len(params)):
        plus, minus = params.copy(), params.copy()
        plus[i] += shift
        minus[i] -= shift
        grad[i] = 0.5 * (energy(ansatz, observable, plus) - energy(ansatz, observable, minus))
    return grad


def run_scenario(name: str, ansatz, *, initial_params: np.ndarray) -> tuple[bool, int]:
    """Run one scenario to completion or early stop.

    Returns:
        (stopped_early, steps_taken)
    """
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    observable = _observable(ansatz.num_qubits)
    monitor = QMLMonitor(
        detectors=_detectors(),
        policy="stop",
        window_size=50,
        planned_steps=PLANNED_STEPS,
    )
    adapter = QiskitAdapter(
        monitor, ansatz, optimizer_name="GradientDescent", learning_rate=LEARNING_RATE
    )
    params = initial_params

    diagnosis = None
    stopped_early = False
    for step in range(MAX_DEMO_STEPS):
        loss = energy(ansatz, observable, params)
        gradients = parameter_shift_gradient(ansatz, observable, params)
        diagnosis = adapter.record_step(step, loss, gradients, parameters=params)
        params = params - LEARNING_RATE * gradients

        if monitor.should_stop():
            stopped_early = True
            break

    steps_taken = monitor.state.step_count
    print(f"Steps taken: {steps_taken} (planned budget: {PLANNED_STEPS})")
    print(f"Final diagnosis: {diagnosis.issue.value} (confidence={diagnosis.confidence:.2f})")
    print(f"Stopped early: {stopped_early}")

    if stopped_early:
        mean_wall_time = monitor.state.mean_wall_time() or 0.0
        remaining_steps = max(0, PLANNED_STEPS - steps_taken)
        estimated_saved = remaining_steps * mean_wall_time
        print(
            f"Estimated compute saved: ~{estimated_saved:.2f}s of wall-clock "
            f"time (extrapolating {remaining_steps} unrun planned steps at "
            f"~{mean_wall_time * 1000:.2f}ms/step)"
        )

    return stopped_early, steps_taken


def main() -> None:
    healthy_stopped, healthy_steps = run_scenario(
        "WITHOUT any issue -- healthy convergence (must NOT stop early)",
        healthy_ansatz(),
        initial_params=np.array([0.9, -0.6, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    )
    plateau_stopped, plateau_steps = run_scenario(
        "WITH a collapsed-gradient run -- possible barren plateau (should stop early)",
        plateau_ansatz(),
        initial_params=np.array([0.3, 0.5]),
    )

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    print(f"Healthy run stopped early:  {healthy_stopped} (expected: False)")
    print(f"Plateau run stopped early: {plateau_stopped} (expected: True)")

    assert not healthy_stopped, "the healthy convergence run should never be stopped early"
    assert plateau_stopped, "the engineered plateau run should be stopped early"
    assert plateau_steps < healthy_steps, "the plateau run should stop well before a full budget"
    print("\nBoth expectations hold: this is the demonstration blueprint Volume XX asks for.")


if __name__ == "__main__":
    main()
