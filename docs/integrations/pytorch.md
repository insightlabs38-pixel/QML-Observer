# PyTorch integration

**Requires:** `pip install qml-observer[torch]` (`torch>=2.0`)

`PyTorchAdapter` (`qml_observer.adapters.pytorch.adapter`) observes a
hybrid quantum-classical training loop built as an ordinary
`torch.nn.Module` (e.g. wrapping a quantum circuit with
`qml.qnn.TorchLayer` or a hand-rolled `torch.autograd.Function`) trained
with a `torch.optim.Optimizer`. Consistent with the project's core
architectural rule, it never calls `.backward()` or reimplements the
optimizer step -- it only observes an already-run one.

## Minimal example

```python
import torch
from qml_observer import QMLMonitor
from qml_observer.adapters.pytorch.adapter import PyTorchAdapter

model = build_hybrid_qnn()  # any torch.nn.Module, e.g. wrapping a qml.qnn.TorchLayer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

monitor = QMLMonitor(policy="log")
adapter = PyTorchAdapter(monitor, module=model, optimizer=optimizer)

for step in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    diagnosis = adapter.record_step(step, loss)
    optimizer.step()
    if monitor.should_stop():
        break
```

## What's automatically extracted

- **Gradients and parameters** -- when a module is attached,
  `record_step()` collects every parameter's `.grad` (flattened and
  concatenated into one 1-D array) automatically after `loss.backward()`;
  parameters with no gradient yet (e.g. an unused branch) are skipped
  rather than raising. Pass `gradients=`/`parameters=` explicitly to
  override auto-collection for a given step.
- **Parameter count** -- `sum(p.numel() for p in module.parameters())`,
  populating `CircuitMetadata.n_parameters`.
- **Optimizer metadata** -- when an optimizer is attached, its class name
  (e.g. `"Adam"`) and `param_groups[0]["lr"]` populate
  `OptimizerMetadata.name`/`learning_rate` automatically. Pass
  `optimizer_name=`/`learning_rate=` to the constructor to set these
  without attaching an optimizer object, or to override what's read from
  one.
- **Gradient method** -- recorded as `"backprop"` by default (PyTorch's
  own autograd), overridable per step via `gradient_method=`.

## Tensor conversion

Loss/gradient/parameter tensors are converted from `torch.Tensor` (still
carrying an autograd graph, on any device) to plain `numpy.ndarray` via
`.detach().cpu().numpy()` before reaching `QMLMonitor.update()` -- see
`qml_observer.adapters.autograd.to_numpy()`, the same conversion the
`JAXAdapter` and standalone `AutogradAdapter` use.

## Attach/detach lifecycle

```python
adapter = PyTorchAdapter(monitor)          # no module/optimizer yet
adapter.attach(module=model)               # attach later
adapter.attach(optimizer=optimizer)        # attach independently
adapter.attached                            # -> True
adapter.detach()                            # clears both
```

`attach()`/`detach()` mirror the `PennyLaneAdapter`/`QiskitAdapter`
lifecycle -- `record_step()` still works with nothing attached, just
without auto-collection or optimizer metadata.
