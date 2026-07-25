"""Milestone 8, Issue #61: Qiskit callback-integration example (Issue #59).

Unlike `basic_monitor.py`/`barren_plateau_demo.py`, which drive a manual
training loop, this example wires `QiskitAdapter.callback` directly into a
real `qiskit-machine-learning` trainer's own `callback=` hook -- the
integration path Volume X's `callback(iteration, parameters, loss)` sketch
exists for, and the one real Qiskit training code most commonly uses in
practice (`VQC`, `NeuralNetworkClassifier`, `NeuralNetworkRegressor`, and
`qiskit-machine-learning`'s own `SPSA`/`COBYLA` optimizers all call back
this way -- see `QiskitAdapter.callback`'s docstring for the exact shapes
handled).

No manual `record_step()` calls happen here at all: `VQC.fit()` drives the
whole loop internally, calling `adapter.callback(weights, obj_func_eval)`
after every objective-function evaluation.

Run with:
    python examples/qiskit/vqc_callback_demo.py
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import efficient_su2, zz_feature_map
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA

from qml_observer import QMLMonitor
from qml_observer.adapters.qiskit.adapter import QiskitAdapter

N_FEATURES = 2
N_SAMPLES = 40
MAX_ITER = 60


def make_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A simple, easily-learnable binary XOR-like dataset."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(N_SAMPLES, N_FEATURES))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    return x, y


def main() -> None:
    monitor = QMLMonitor(policy="log", window_size=50)

    feature_map = zz_feature_map(N_FEATURES)
    ansatz = efficient_su2(N_FEATURES, reps=1)
    optimizer = COBYLA(maxiter=MAX_ITER)

    # QiskitAdapter starts unattached; VQC.circuit only exists once VQC is
    # constructed, so we attach it right after (Issue #58's attach/detach
    # lifecycle -- same two-step pattern PennyLaneAdapter uses when the
    # QNode isn't available yet at adapter-construction time).
    adapter = QiskitAdapter(monitor, optimizer=optimizer)

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=adapter.callback,
    )
    adapter.attach(vqc)

    X, y = make_dataset()
    print(f"Run ID: {monitor.run_id}")
    print(f"Training VQC on {N_SAMPLES} samples, optimizer={type(optimizer).__name__}\n")

    vqc.fit(X, y)

    final = monitor.finish()
    print(f"Total callback-driven steps recorded: {monitor.state.step_count}")
    print(f"Final diagnosis: {final.issue.value} (confidence={final.confidence:.2f})")

    train_accuracy = vqc.score(X, y)
    print(f"Training accuracy: {train_accuracy:.2%}")

    latest_optimizer_meta = monitor.state.latest_observation.optimizer
    if latest_optimizer_meta is not None:
        print(
            f"Optimizer metadata recorded via normalize_optimizer_metadata(): "
            f"name={latest_optimizer_meta.name!r}, "
            f"gradient_method={latest_optimizer_meta.gradient_method!r}"
        )


if __name__ == "__main__":
    main()
