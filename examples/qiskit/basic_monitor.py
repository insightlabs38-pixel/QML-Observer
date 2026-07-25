"""Milestone 8, Issue #61: basic Qiskit + QMLMonitor example.

The smallest possible integration: attach a `QiskitAdapter` to a
`QuantumCircuit` ansatz, run a manual training loop computing loss and
gradients via Qiskit's own `Estimator` primitive (parameter-shift, applied
by hand -- the adapter never reimplements this, only observes the
result), and print the diagnosis each step. No detectors are configured
here on purpose, mirroring `examples/pennylane/basic_monitor.py`: this
script exists to prove the plumbing (QuantumCircuit -> adapter ->
QMLMonitor -> DiagnosisResult) works end to end with minimal code.

For a version with real detectors wired in, see `barren_plateau_demo.py`.
For the Qiskit-specific callback-integration path (Issue #59), see
`vqc_callback_demo.py`.

Run with:
    python examples/qiskit/basic_monitor.py
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from qml_observer import QMLMonitor
from qml_observer.adapters.qiskit.adapter import QiskitAdapter

N_QUBITS = 2
N_STEPS = 30
LEARNING_RATE = 0.4

_estimator = StatevectorEstimator()
_observable = SparsePauliOp("Z" + "I" * (N_QUBITS - 1))


def build_ansatz():
    return efficient_su2(N_QUBITS, reps=1)


def energy(ansatz, params: np.ndarray) -> float:
    """Expectation value of `_observable`, evaluated at `params`."""
    result = _estimator.run([(ansatz, _observable, [params])]).result()
    return float(result[0].data.evs[0])


def parameter_shift_gradient(ansatz, params: np.ndarray) -> np.ndarray:
    """Textbook two-term parameter-shift rule (shift = pi/2), valid for the
    single-qubit Pauli rotation gates `efficient_su2` is built from. This
    lives in the example, not the adapter: per the blueprint, the adapter
    only *observes* already-computed gradients, it never computes them."""
    grad = np.zeros_like(params)
    shift = np.pi / 2
    for i in range(len(params)):
        plus = params.copy()
        plus[i] += shift
        minus = params.copy()
        minus[i] -= shift
        grad[i] = 0.5 * (energy(ansatz, plus) - energy(ansatz, minus))
    return grad


def main() -> None:
    ansatz = build_ansatz()

    # No detectors configured: every step reports the INSUFFICIENT_EVIDENCE
    # placeholder. That's expected here -- this script is about the
    # integration plumbing, not detection. policy="log" keeps output quiet
    # (no terminal ALERT banners) since there's nothing to warn about yet.
    monitor = QMLMonitor(policy="log")
    adapter = QiskitAdapter(
        monitor, ansatz, optimizer_name="GradientDescent", learning_rate=LEARNING_RATE
    )

    rng = np.random.default_rng(0)
    params = rng.uniform(-0.5, 0.5, size=ansatz.num_parameters)

    print(f"Run ID: {monitor.run_id}\n")
    for step in range(N_STEPS):
        loss = energy(ansatz, params)
        gradients = parameter_shift_gradient(ansatz, params)
        diagnosis = adapter.record_step(step, loss, gradients, parameters=params)
        print(f"step={step:>2}  loss={loss: .4f}  issue={diagnosis.issue.value}")
        params = params - LEARNING_RATE * gradients

    final = monitor.finish()
    print(f"\nFinal diagnosis: {final.issue.value} (confidence={final.confidence:.2f})")
    print(f"Total steps recorded: {monitor.state.step_count}")


if __name__ == "__main__":
    main()
