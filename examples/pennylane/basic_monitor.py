"""Milestone 6, Issue #46: basic PennyLane + QMLMonitor example.

The smallest possible integration: attach a `PennyLaneAdapter` to a QNode,
run a manual training loop, and print the diagnosis each step. No
detectors are configured here on purpose -- this script exists to prove
the plumbing (QNode -> adapter -> QMLMonitor -> DiagnosisResult) works
end to end with minimal code, matching the blueprint's Week 1 deliverable:

    Generic training loop
            |
        QMLMonitor
            |
        TrainingEvent
            |
         (diagnosis)

For a version with real detectors wired in (and the "stop before wasting
compute" behavior this project is actually for), see
`barren_plateau_demo.py`.

Run with:
    python examples/pennylane/basic_monitor.py
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

from qml_observer import QMLMonitor
from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter

N_WIRES = 2
N_STEPS = 30


def build_circuit() -> qml.QNode:
    dev = qml.device("default.qubit", wires=N_WIRES, shots=None)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    return circuit


def main() -> None:
    circuit = build_circuit()

    # No detectors configured: every step reports the INSUFFICIENT_EVIDENCE
    # placeholder. That's expected here -- this script is about the
    # integration plumbing, not detection. policy="log" keeps output quiet
    # (no terminal ALERT banners) since there's nothing to warn about yet.
    monitor = QMLMonitor(policy="log")
    adapter = PennyLaneAdapter(
        monitor, circuit, optimizer_name="GradientDescent", learning_rate=0.4
    )

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    params = pnp.array([0.9, -0.6], requires_grad=True)

    print(f"Run ID: {monitor.run_id}\n")
    for step in range(N_STEPS):
        gradients = qml.grad(circuit)(params)
        loss = circuit(params)
        diagnosis = adapter.record_step(step, float(loss), gradients, parameters=params)
        print(f"step={step:>2}  loss={float(loss): .4f}  issue={diagnosis.issue.value}")
        params = opt.step(circuit, params)

    final = monitor.finish()
    print(f"\nFinal diagnosis: {final.issue.value} (confidence={final.confidence:.2f})")
    print(f"Total steps recorded: {monitor.state.step_count}")


if __name__ == "__main__":
    main()
