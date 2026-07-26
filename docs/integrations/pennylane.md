# PennyLane integration

**Requires:** `pip install qml-observer[pennylane]` (`pennylane>=0.35`)

`PennyLaneAdapter` (`qml_observer.adapters.pennylane.adapter`) *observes*
your PennyLane training -- it never reimplements PennyLane's own gradient
machinery. Your loop still computes `loss`/`gradients` however it likes
(autograd/Torch/JAX interfaces; parameter-shift, adjoint, finite-diff, or
any other `diff_method`); the adapter's job is only to forward
already-computed values to `QMLMonitor.update()` and auto-populate
`CircuitMetadata`/`OptimizerMetadata` from the attached `QNode`.

## Minimal example

```python
import pennylane as qml
from pennylane import numpy as pnp
from qml_observer import QMLMonitor
from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter

dev = qml.device("default.qubit", wires=2, shots=None)


@qml.qnode(dev, diff_method="parameter-shift")
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[1], wires=1)
    return qml.expval(qml.PauliZ(1))


monitor = QMLMonitor(policy="log")
adapter = PennyLaneAdapter(monitor, circuit, optimizer_name="GradientDescent", learning_rate=0.4)

opt = qml.GradientDescentOptimizer(stepsize=0.4)
params = pnp.array([0.9, -0.6], requires_grad=True)

for step in range(30):
    gradients = qml.grad(circuit)(params)
    loss = circuit(params)
    diagnosis = adapter.record_step(step, float(loss), gradients, parameters=params)
    params = opt.step(circuit, params)
```

Add real detectors (`BarrenPlateauDetector`, `StagnationDetector`,
`ConvergenceDetector`) and `policy="stop"` to get the full "stop before
wasting compute" behavior -- see
`examples/pennylane/barren_plateau_demo.py` for the complete, runnable
version of the blueprint's "critical MVP demo".

## What's automatically extracted

- **Gradient method** -- the `QNode`'s configured `diff_method`, verified
  for both `"parameter-shift"` and `"adjoint"`, recorded into
  `OptimizerMetadata.gradient_method`.
- **Shot count** -- inferred from the constructed tape for finite-shots
  devices (falling back to the device default), `None` for analytic
  (`shots=None`) circuits. An explicit `shots=` argument to
  `record_step()` always overrides inference.
- **Circuit metadata** -- qubit count, depth, gate count, entangling-gate
  count, and parameter count, extracted defensively from the PennyLane
  tape/`QuantumScript` so an unexpected PennyLane version/tape shape
  degrades to `None` fields instead of raising.

## Finite shots

Finite-shots training (`shots=N` on the device) works the same way --
gradients will simply be noisier, which is exactly the scenario
`examples/pennylane/noisy_training.py` demonstrates without triggering a
false-positive plateau detection.

## Version compatibility

The project targets `pennylane>=0.35`. Because PennyLane's internal tape
APIs have changed across releases, every piece of metadata extraction here
is best-effort: if a given version doesn't expose something the way
expected, that field is simply left `None` rather than raising --
consistent with the project's fail-open policy (addendum §1).
