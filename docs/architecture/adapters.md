# Adapters

Adapters (`qml_observer.adapters`) are the only layer allowed to import a
specific framework. Each one converts framework-specific objects into the
shared event schema and calls `QMLMonitor.update()` -- they never
reimplement a framework's own gradient machinery or optimization loop,
only observe already-computed results.

- **`GenericAdapter`** (`adapters/generic.py`) -- the lowest-level,
  framework-neutral path: a thin, explicit pass-through to
  `monitor.update()` for fully custom research training loops that
  already produce plain `float`/`numpy.ndarray` values.
- **`AutogradAdapter`** (`adapters/autograd.py`) -- the framework-neutral
  path for a custom loop computing `loss`/`gradients`/`parameters` with
  *some* classical autodiff library instead of plain arrays: duck-types
  `.detach()`/`.cpu()`/`.numpy()`-style conversion (falling back to
  `np.asarray()`) so any framework's tensors reach `monitor.update()`
  safely. Requires no extra dependency of its own, and is the shared base
  class both `PyTorchAdapter` and `JAXAdapter` build on. See
  `integrations/generic.md`.
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
- **`PyTorchAdapter`** (`adapters/pytorch/adapter.py`, Milestone 14) --
  `attach()`/`detach()` a `torch.nn.Module`/`torch.optim.Optimizer`;
  `record_step()` auto-collects gradients/parameters from the attached
  module after `loss.backward()` and reads optimizer name/learning rate
  off the attached optimizer. Requires the `torch` extra. See
  `integrations/pytorch.md`.
- **`JAXAdapter`** (`adapters/jax/adapter.py`, Milestone 14) -- `attach()`/
  `detach()` a parameter pytree; `record_step()` flattens
  gradient/parameter pytrees (via `jax.tree_util.tree_leaves`) into the
  1-D array `monitor.update()` expects. Requires the `jax` extra. See
  `integrations/jax.md`.

Every adapter's `record_step()`/`record()`/`callback()` ultimately funnels
into the same `QMLMonitor.update()` call, which is why the detection/
diagnosis/action/reporting layers behind it are identical regardless of
which adapter (or none) is driving them.

## Experiment-tracker integrations

Alongside adapters (which feed *into* `QMLMonitor`), Milestone 14 also
added integrations on the *output* side: `MLflowTracker`/`WandbTracker`
(`integrations/trackers/`) implement the same `RunReporter` duck type
`QMLMonitor(reporter=...)` drives, forwarding events/diagnoses into an
existing MLflow/W&B run instead of only JSONL. See
`integrations/experiment_trackers.md`.
