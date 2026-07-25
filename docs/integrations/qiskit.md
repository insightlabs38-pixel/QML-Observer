# Qiskit integration

`qml_observer.adapters.qiskit.adapter.QiskitAdapter` is QML Observer's
second real framework integration (Milestone 8), covering:

- manual training loops built on a `QuantumCircuit` + `Estimator`/`Sampler`
  primitive,
- `qiskit_algorithms`-style optimizers with native callback support
  (e.g. `SPSA`), and
- `qiskit-machine-learning` trainers (`VQC`, `NeuralNetworkClassifier`,
  `NeuralNetworkRegressor`).

Like `PennyLaneAdapter`, it **observes** training rather than owning any
part of it: your code still computes the loss, gradients, and optimizer
steps. The adapter's job is to forward that already-computed information
into `QMLMonitor.update()` and to fill in `CircuitMetadata`/
`OptimizerMetadata` automatically wherever Qiskit exposes enough
information to do so.

## Installation

```bash
pip install qml-observer[qiskit]
# or, for local development:
pip install -e ".[dev,qiskit]"
```

This installs `qiskit>=1.0` and `qiskit-machine-learning>=0.7`.
Constructing a `QiskitAdapter` without the `qiskit` package installed
raises a clear `ImportError` rather than failing on first use.

## Quickstart: manual training loop

The lowest-level integration path — you compute loss and gradients
yourself (e.g. via parameter-shift against an `Estimator`), and call
`record_step()` once per iteration:

```python
from qiskit.circuit.library import efficient_su2

from qml_observer import QMLMonitor
from qml_observer.adapters.qiskit.adapter import QiskitAdapter

ansatz = efficient_su2(4, reps=2)
monitor = QMLMonitor()
adapter = QiskitAdapter(monitor, ansatz, optimizer_name="COBYLA")

for step in range(200):
    loss, gradients = my_energy_and_gradient(params)
    diagnosis = adapter.record_step(step, loss, gradients, parameters=params)
    if monitor.should_stop():
        break
```

Passing `circuit` to the constructor (as above) is equivalent to calling
`adapter.attach(ansatz)` immediately afterward; `record_step()` then
auto-populates `CircuitMetadata` on every call from the attached circuit.
Call `adapter.detach()` to stop this (subsequent `record_step()` calls
still work, just without automatic circuit metadata).

If your gradient is computed in a separate step from your loss (common
with a standalone `BaseEstimatorGradient`), cache it with
`record_gradient()` and it will be picked up by the next `record_step()`
call automatically:

```python
adapter.record_gradient(gradients)
diagnosis = adapter.record_step(step, loss, parameters=params)
```

## Callback integration

`QiskitAdapter.callback` can be passed directly as the `callback=`
argument to anything that calls back with one of the shapes Qiskit's
ecosystem actually uses in practice — no manual training loop required.

**`qiskit_algorithms` optimizers with native callback support** (e.g.
`SPSA`), which call back as `callback(nfev, params, fval, stepsize, accepted)`:

```python
from qiskit_algorithms.optimizers import SPSA

adapter = QiskitAdapter(monitor, ansatz, optimizer=SPSA(maxiter=200))
opt = SPSA(maxiter=200, callback=adapter.callback)
result = opt.minimize(cost_fn, x0)
```

**`qiskit-machine-learning` trainers** (`VQC`, `NeuralNetworkClassifier`,
`NeuralNetworkRegressor`), which call back as `callback(weights, obj_func_eval)`:

```python
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA

adapter = QiskitAdapter(monitor)
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100),
    callback=adapter.callback,
)
adapter.attach(vqc)  # reads vqc.circuit / vqc.ansatz for CircuitMetadata
vqc.fit(X, y)
```

`callback()` also accepts plain `scipy.optimize.minimize`-style
callbacks (`callback(xk)`, no loss reported) and the blueprint's own
manual 3-argument form (`callback(iteration, parameters, loss)`). The
shape is detected purely by positional argument count — see the
`QiskitAdapter` module docstring for the full mapping. An unrecognized
argument count raises `TypeError` immediately rather than silently
misinterpreting the callback.

Because callback-driven integrations don't supply an iteration index,
`QiskitAdapter` auto-increments one internally (starting at 0) whenever
a callback shape doesn't include an explicit iteration/`nfev` argument.

## Circuit metadata extraction

`extract_circuit_metadata()` builds a `CircuitMetadata` from any
`qiskit.circuit.QuantumCircuit` (an ansatz, or a full trainer circuit
including a feature map): qubit count, depth, parameter count, gate
count, and entangling-gate count. Every field is extracted defensively —
if an installed Qiskit version doesn't expose something the way this
method expects, that field is left `None` rather than raising, matching
`CircuitMetadata`'s "every field but the essentials is optional" design.

`ansatz_name` and `initialization` are **not** auto-detected (a generic
`QuantumCircuit` doesn't generically expose either); pass them explicitly
to `record_step()`/`extract_circuit_metadata()` if you want them recorded.

## Optimizer metadata normalization

Qiskit optimizer objects expose their configuration inconsistently
across classes: `SPSA` uses a `"learning_rate"` key in `.settings`, `ADAM`
uses `"lr"`, and gradient-free optimizers (`COBYLA`, `NELDER_MEAD`,
`POWELL`) expose no learning rate at all. `normalize_optimizer_metadata()`
handles this best-effort via each optimizer's own `.settings` dict rather
than hardcoding attribute access per class, and never raises on an
unrecognized optimizer type. It also infers a `gradient_method` label
(e.g. `"spsa-approximation"`, `"gradient-free"`, `"finite-difference"`)
from a small, conservative table of known optimizer class names — an
unrecognized class simply leaves `gradient_method=None`.

Any of `optimizer_name=`, `learning_rate=`, or `gradient_method=` passed
explicitly to `QiskitAdapter.__init__()` always takes precedence over
whatever would otherwise be inferred from a live `optimizer=` object.

## Version variance

Per the blueprint's Volume X guidance ("because Qiskit APIs vary across
components, isolate version-specific logic inside this adapter"), all
version-specific handling lives inside `adapters/qiskit/adapter.py`
rather than leaking into the core monitor/detector layers. In practice
that means:

- Callback signature detection (`_normalize_callback_args`) is based on
  positional argument *count*, not on `isinstance` checks against a
  specific optimizer/trainer class, so it degrades gracefully across
  `qiskit_algorithms` and `qiskit-machine-learning` versions that share
  the same callback shapes.
- Optimizer metadata extraction reads `.settings` dicts rather than
  fixed attribute names, and every circuit-introspection call is wrapped
  so an unexpected Qiskit version simply yields a `None` field instead of
  an exception (consistent with the project's fail-open policy, addendum
  §1 — `QMLMonitor.update()` itself also fail-opens around the whole
  step, so even a totally unanticipated internal change degrades to a
  `degraded=True` diagnosis rather than crashing your training loop).

## Examples

- [`examples/qiskit/basic_monitor.py`](../../examples/qiskit/basic_monitor.py)
  — minimal integration via a manual training loop and hand-rolled
  parameter-shift gradients, no detectors configured.
- [`examples/qiskit/barren_plateau_demo.py`](../../examples/qiskit/barren_plateau_demo.py)
  — the Qiskit version of the project's flagship demo: a healthy run that
  completes normally, contrasted with an engineered collapsed-gradient
  run that is stopped early with an estimated compute-saved figure.
- [`examples/qiskit/vqc_callback_demo.py`](../../examples/qiskit/vqc_callback_demo.py)
  — wires `QiskitAdapter.callback` directly into a real
  `qiskit-machine-learning` `VQC` trainer's own `callback=` hook, with no
  manual training loop at all.

## Tests

- `tests/unit/adapters/test_qiskit.py` — unit coverage for `attach()`/
  `detach()`, `record_step()`/`record_gradient()`, all four `callback()`
  argument shapes, `extract_circuit_metadata()`, and
  `normalize_optimizer_metadata()`.
- `tests/integration/qiskit/test_qiskit_end_to_end.py` — end-to-end tests
  driving a real `QuantumCircuit`, a real Qiskit `Estimator` primitive,
  and a real `VQC.fit()` call through the full
  adapter → monitor → detector → diagnosis → action pipeline.

## See also

The PennyLane and generic-adapter integration guides referenced by the
blueprint (`docs/integrations/pennylane.md`, `docs/integrations/generic.md`)
have not been written yet; until they land, see
[`src/qml_observer/adapters/pennylane/adapter.py`](../../src/qml_observer/adapters/pennylane/adapter.py)
and [`src/qml_observer/adapters/generic.py`](../../src/qml_observer/adapters/generic.py)
for the equivalent docstring-level documentation.
