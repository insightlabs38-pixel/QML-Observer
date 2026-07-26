# Generic integration (any framework, or none)

**Requires:** nothing extra -- this is the core package's own integration
surface, with no PennyLane/Qiskit dependency.

`GenericAdapter` (`qml_observer.adapters.generic`) is the framework-neutral
path every other adapter effectively sits in front of: it exposes the same
plain-argument shape (`step`, `loss`, `gradients`, `parameters`, `circuit`,
`optimizer`, `shots`) that `PennyLaneAdapter`/`QiskitAdapter` translate
their framework-specific objects into. Use this directly if you have a
fully custom research training loop (TensorFlow, JAX, a from-scratch
simulator, etc.) and don't want to wait on a framework-specific adapter.

## Example

```python
from qml_observer import QMLMonitor
from qml_observer.adapters.generic import GenericAdapter
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.stagnation import StagnationDetector
from qml_observer.detectors.convergence import ConvergenceDetector

monitor = QMLMonitor(
    detectors=[BarrenPlateauDetector(), StagnationDetector(), ConvergenceDetector()],
    policy="stop",
)
adapter = GenericAdapter(monitor)

for step in range(10_000):
    loss, gradients = my_training_step()  # your own framework/simulator
    diagnosis = adapter.record(step, loss=loss, gradients=gradients)
    if monitor.should_stop():
        break
```

`GenericAdapter.record()` is a direct pass-through to
`QMLMonitor.update()` -- see that method's docstring for full argument
semantics, including the fail-open guarantee (step-processing errors never
raise here; only misuse, like calling `record()` after the run has
`finish()`-ed, does).

Everything downstream of this call -- detectors, diagnosis, actions,
reporting, the CLI -- behaves identically regardless of whether it was
reached through `GenericAdapter`, `PennyLaneAdapter`, or `QiskitAdapter`.
If you'd rather skip the adapter object entirely, you can call
`monitor.update(...)` directly with the same arguments; `GenericAdapter`
exists only to give you a named, documented integration surface matching
the project's adapter-layer vocabulary.
