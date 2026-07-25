"""Milestone 6, Issue #46: the "critical MVP demo" (blueprint Volume XX).

Runs two training scenarios back to back with real detectors wired in and
`policy="stop"`:

  1. A healthy, well-conditioned circuit that converges normally --
     it must run to completion and must NOT be stopped early.
  2. An engineered plateau-like circuit -- deliberately built from only
     RZ rotations measured in the Z basis, so its gradient is
     (numerically) exactly zero from step 0 regardless of qubit/depth
     scaling. This is a stand-in for an actual barren-plateau-afflicted
     ansatz: it produces the same observable signature (collapsed
     gradient + stagnant loss) without needing the ~15-20+ qubits a real
     random-circuit barren plateau needs to reproduce reliably, which
     would make this example too slow to run as a quick demo.

The second run should be caught and stopped well before it would
otherwise reach its planned step budget, and the script reports the
estimated compute saved using the addendum's resolved formula:

    saved = (planned_total_steps - actual_steps_at_stop) * mean_wall_time_per_step

This mirrors the blueprint's central early-adoption pitch: a healthy run
is never incorrectly terminated, and a genuinely unproductive run is
stopped early with a clear, explainable diagnosis.

Run with:
    python examples/pennylane/barren_plateau_demo.py
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as pnp

from qml_observer import QMLMonitor
from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.convergence import ConvergenceDetector
from qml_observer.detectors.stagnation import StagnationDetector

PATIENCE = 15
PLANNED_STEPS = 5_000  # the compute budget this run would otherwise be given
MAX_DEMO_STEPS = 200  # hard cap so the demo itself can't run away


def _detectors() -> list:
    """A fresh detector set per run -- detectors are stateful, so each
    `QMLMonitor` needs its own instances rather than a shared list."""
    return [
        BarrenPlateauDetector(patience=PATIENCE),
        StagnationDetector(patience=PATIENCE),
        ConvergenceDetector(patience=PATIENCE, loss_threshold=1e-2),
    ]


def healthy_circuit() -> qml.QNode:
    """A simple, well-conditioned 2-qubit ansatz that converges normally."""
    dev = qml.device("default.qubit", wires=2, shots=None)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    return circuit


def plateau_circuit() -> qml.QNode:
    """An engineered plateau: RZ-only on |0>, Z-basis measurement -> the
    expectation value (and therefore the gradient) is invariant to every
    parameter, exactly the "gradient collapse + stagnant loss" signature
    a real barren plateau produces."""
    dev = qml.device("default.qubit", wires=2, shots=None)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        qml.RZ(params[0], wires=0)
        qml.RZ(params[1], wires=1)
        return qml.expval(qml.PauliZ(0))

    return circuit


def run_scenario(name: str, circuit: qml.QNode, *, initial_params) -> tuple[bool, int]:
    """Run one scenario to completion or early stop.

    Returns:
        (stopped_early, steps_taken)
    """
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    monitor = QMLMonitor(
        detectors=_detectors(),
        policy="stop",
        window_size=50,
        planned_steps=PLANNED_STEPS,
    )
    adapter = PennyLaneAdapter(
        monitor, circuit, optimizer_name="GradientDescent", learning_rate=0.4
    )
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    params = initial_params

    diagnosis = None
    stopped_early = False
    for step in range(MAX_DEMO_STEPS):
        gradients = qml.grad(circuit)(params)
        loss = circuit(params)
        diagnosis = adapter.record_step(step, float(loss), gradients, parameters=params)
        params = opt.step(circuit, params)

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
        healthy_circuit(),
        initial_params=pnp.array([0.9, -0.6], requires_grad=True),
    )
    plateau_stopped, plateau_steps = run_scenario(
        "WITH a collapsed-gradient run -- possible barren plateau (should stop early)",
        plateau_circuit(),
        initial_params=pnp.array([0.3, 0.5], requires_grad=True),
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
