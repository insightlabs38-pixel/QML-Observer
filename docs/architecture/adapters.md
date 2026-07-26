# Adapters

Adapters (`qml_observer.adapters`) are the only layer allowed to import a
specific framework. Each one converts framework-specific objects into the
shared event schema and calls `QMLMonitor.update()` -- they never
reimplement a framework's own gradient machinery or optimization loop,
only observe already-computed results.

- **`GenericAdapter`** (`adapters/generic.py`) -- the lowest-level,
  framework-neutral path: a thin, explicit pass-through to
  `monitor.update()` for fully custom research training loops.
- **`PennyLaneAdapter`** (`adapters/pennylane/adapter.py`) -- `attach()`/
  `detach()` a `QNode`; `record_step()` observes already-computed
  loss/gradients/parameters and auto-populates `CircuitMetadata`/
  `OptimizerMetadata`. Supports parameter-shift, adjoint differentiation,
  and both analytic and finite-shots execution. Requires the `pennylane`
  extra. See `integrations/pennylane.md`.
- **`QiskitAdapter`** (`adapters/qiskit/adapter.py`) -- `attach()`/
  `detach()` a `QuantumCircuit` or a trainer exposing one (`VQC`,
  `NeuralNetworkClassifier`); `record_step()`/`record_gradient()` observe
  results, and `callback()` normalizes across the several optimizer/trainer
  callback shapes seen in practice (`qiskit-machine-learning` trainer
  style, SPSA-style, plain `scipy.optimize.minimize`-style, and the
  blueprint's manual `(iteration, parameters, loss)` form) so it can be
  passed directly as `callback=adapter.callback`. Requires the `qiskit`
  extra. See `integrations/qiskit.md`.

Every adapter's `record_step()`/`record()`/`callback()` ultimately funnels
into the same `QMLMonitor.update()` call, which is why the detection/
diagnosis/action/reporting layers behind it are identical regardless of
which adapter (or none) is driving them.
