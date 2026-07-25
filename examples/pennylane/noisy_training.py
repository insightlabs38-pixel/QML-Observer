"""Milestone 6, Issue #46: shot-noise-dominated training example.

Trains the same simple ansatz twice: once with analytic (`shots=None`)
gradients, once with a small finite-shots budget. This is *not* a
NoiseDetector demo -- that detector is Milestone 9 (Issue #66) and
doesn't exist yet. What this script demonstrates instead, using only
statistics that are actually implemented today (Milestone 3:
`gradient_norm`, `gradient_variance`):

  * Per-step gradient statistics are visibly noisier under a small shot
    budget than under analytic simulation, even though both are
    converging toward the same optimum.
  * The already-implemented detectors (`BarrenPlateauDetector`,
    `StagnationDetector`) require a *persistent*, multi-step condition
    (see `patience`) before triggering, precisely so that noisy-but-real
    single-step gradients don't get misdiagnosed as a plateau. This is
    the mechanism the addendum's calibration methodology (§3) exists to
    tune -- and the future gap this example intentionally leaves open
    is *quantifying* the shot budget against measurement uncertainty
    (`estimate_measurement_uncertainty`, `NoiseDetector`), which needs
    the Milestone 9 statistics/detector work this example is not
    substituting for.

Run with:
    python examples/pennylane/noisy_training.py
"""

from __future__ import annotations

import warnings

import pennylane as qml
from pennylane import numpy as pnp

from qml_observer import QMLMonitor
from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.convergence import ConvergenceDetector
from qml_observer.detectors.stagnation import StagnationDetector
from qml_observer.statistics.gradients import gradient_norm, gradient_variance

# Some PennyLane releases in the supported range (>=0.35) deprecate passing
# `shots=` directly to `qml.device(...)` in favor of a `set_shots` transform.
# Both spellings still work; silence the notice so this demo's output stays
# focused on the diagnosis, not a version-specific API nudge.
warnings.filterwarnings("ignore", message=".*Setting shots on device is deprecated.*")

N_STEPS = 40
PATIENCE = 15
PRINT_EVERY = 10


def build_circuit(shots: int | None) -> qml.QNode:
    dev = qml.device("default.qubit", wires=2, shots=shots)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[1], wires=1)
        return qml.expval(qml.PauliZ(1))

    return circuit


def _detectors() -> list:
    return [
        BarrenPlateauDetector(patience=PATIENCE),
        StagnationDetector(patience=PATIENCE),
        ConvergenceDetector(patience=PATIENCE, loss_threshold=1e-2),
    ]


def run(label: str, shots: int | None) -> None:
    print(f"\n{'=' * 60}\n{label} (shots={shots})\n{'=' * 60}")

    circuit = build_circuit(shots)
    monitor = QMLMonitor(detectors=_detectors(), policy="log", window_size=50)
    adapter = PennyLaneAdapter(
        monitor, circuit, optimizer_name="GradientDescent", learning_rate=0.4
    )
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    params = pnp.array([0.9, -0.6], requires_grad=True)

    diagnosis = None
    for step in range(N_STEPS):
        gradients = qml.grad(circuit)(params)
        loss = circuit(params)
        diagnosis = adapter.record_step(step, float(loss), gradients, parameters=params)
        if step % PRINT_EVERY == 0:
            print(
                f"step={step:>2}  loss={float(loss): .4f}  "
                f"grad_norm={gradient_norm(gradients):.4f}  "
                f"grad_var={gradient_variance(gradients):.2e}  "
                f"issue={diagnosis.issue.value}"
            )
        params = opt.step(circuit, params)

    print(f"Final diagnosis: {diagnosis.issue.value} (confidence={diagnosis.confidence:.2f})")
    print(f"should_stop() ever have fired: {monitor.should_stop()}")


def main() -> None:
    # Fix the RNG so the finite-shots run above is reproducible across runs.
    pnp.random.seed(0)
    run("Analytic (noise-free) simulation", shots=None)
    run("Finite-shots simulation (noisy gradients)", shots=20)

    print(
        f"\n{'=' * 60}\n"
        "Note: both runs use the *same* detectors/thresholds tuned for "
        "analytic simulation. The finite-shots run's per-step gradient "
        "statistics are visibly noisier, but neither run should be "
        "misdiagnosed as a plateau, since a single noisy step never "
        "satisfies the patience-window persistence requirement on its "
        "own. Making that guarantee robust across arbitrary shot budgets "
        "is exactly the Milestone 9 noise-aware diagnostics work.\n"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    main()
