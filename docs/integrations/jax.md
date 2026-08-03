# JAX integration

**Requires:** `pip install qml-observer[jax]` (`jax>=0.4`)

`JAXAdapter` (`qml_observer.adapters.jax.adapter`) observes a hybrid
quantum-classical training loop written in JAX's typical functional
style, where parameters and gradients are **pytrees** (nested
dicts/lists/tuples of arrays) rather than attributes of a stateful
module/optimizer object. Its job over `AutogradAdapter`/`GenericAdapter`
is flattening those pytrees -- via `jax.tree_util.tree_leaves` -- into the
single 1-D array `QMLMonitor.update()` expects, and counting total
parameters across the whole pytree.

## Minimal example

```python
import jax
from qml_observer import QMLMonitor
from qml_observer.adapters.jax.adapter import JAXAdapter

params = init_hybrid_params()  # a pytree of jax arrays
monitor = QMLMonitor(policy="log")
adapter = JAXAdapter(monitor, params, optimizer_name="Adam", learning_rate=0.01)

for step in range(200):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    diagnosis = adapter.record_step(step, loss, grads, params)
    params = optax_update(params, grads)
    if monitor.should_stop():
        break
```

## What's automatically extracted

- **Gradients and parameters** -- any pytree passed as `gradients=`/
  `parameters=` is flattened into one 1-D array via
  `jax.tree_util.tree_leaves`. A single (non-pytree) array works too, the
  same as `AutogradAdapter`.
- **Parameter count** -- summed across every leaf's `.size` in the
  pytree, populating `CircuitMetadata.n_parameters`. If `parameters=`
  isn't passed to a given `record_step()` call, the pytree given to
  `attach()`/the constructor is used as a fallback purely for counting.
- **Gradient method** -- recorded as `"autodiff"` by default, overridable
  per step via `gradient_method=`.

## Optimizer metadata isn't auto-detected

Unlike `PyTorchAdapter`, `JAXAdapter` does **not** introspect an optimizer
object: `optax` optimizer state is itself an opaque pytree with no
standard place to read a name or learning rate from. Pass
`optimizer_name=`/`learning_rate=` explicitly (constructor-only, matching
`PennyLaneAdapter`'s same pattern for the same reason) if you want
`OptimizerMetadata` populated.

## Attach/detach lifecycle

```python
adapter = JAXAdapter(monitor)               # no params template yet
adapter.attach(params)                       # attach a pytree later, for counting
adapter.attached                             # -> True
adapter.detach()                             # clears the template
```
