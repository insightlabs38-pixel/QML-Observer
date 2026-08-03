# Plugin API: writing your own adapters, reporters, and detectors

This page documents how to extend qml_observer with your own **adapter**
(a new framework integration), **reporter/tracker** (a new sink for
`QMLMonitor`'s output), or **detector plugin** (a third-party
`BaseDetector`), using the same contracts
`PennyLaneAdapter`/`QiskitAdapter`/`PyTorchAdapter`/`JAXAdapter`,
`RunReporter`/`MLflowTracker`/`WandbTracker`, and
`qml_observer.detectors.plugins` already implement.

Detector plugins have their own, lighter section near the bottom of this
page (["Detector plugins"](#detector-plugins-issue-103)) since they're
governed differently: a detector participates in the diagnosis/detection
pipeline itself, so it has its own RFC process for *built-in* proposals
(distinct from the plugin mechanism covered here) and its own explicit
"no sandboxing" security note in [`SECURITY.md`](../../SECURITY.md).
Adapters and reporters, covered first below, are a lighter-weight
extension point with no scoring behavior to propose or review.

## Why write an adapter?

An adapter's only job is converting framework-specific (or, for a
tensor/array-based custom framework, merely differently-shaped)
information into calls against `QMLMonitor.update()`. Write one if:

- You have a training loop in a framework with no existing adapter
  (TensorFlow, a research simulator, etc.), *and*
- That framework has enough of its own structure (a stateful
  module/optimizer object, a pytree convention, tape/circuit
  introspection) that a thin, reusable wrapper is worth writing instead of
  calling `monitor.update(...)`/`GenericAdapter`/`AutogradAdapter`
  directly every time.

If neither applies, you likely don't need a new adapter at all -- see
[`integrations/generic.md`](../integrations/generic.md) (plain
`float`/`numpy.ndarray` values) and
[`integrations/generic.md#using-a-classical-autodiff-framework-instead-see-autogradadapter`](../integrations/generic.md)
(any autodiff framework's tensors, via duck typing).

## The adapter contract

There's no `abc.ABC` base class enforcing a single adapter interface --
`GenericAdapter`, `AutogradAdapter`, `PennyLaneAdapter`, `QiskitAdapter`,
`PyTorchAdapter`, and `JAXAdapter` all differ in their public method names
(`record()` vs `record_step()`) and constructor arguments, because each
framework's natural calling convention differs. What they share is a
*shape*, not a fixed ABC:

1. **Wrap a `QMLMonitor`.** Store it (`self.monitor = monitor`), and
   validate it's actually a `QMLMonitor` instance at construction --
   every existing adapter raises `TypeError` immediately rather than
   failing confusingly later.
2. **Expose one recording method** that accepts your framework's natural
   objects (a `torch.nn.Module`'s attached parameters, a PennyLane
   `QNode`'s tape, a JAX pytree, ...) and ends by calling
   `self.monitor.update(step=..., loss=..., gradients=..., parameters=...,
   circuit=..., optimizer=..., shots=...)`, returning whatever
   `update()` returns (a `DiagnosisResult`) unchanged.
3. **Convert, don't reimplement.** Per the blueprint's core architectural
   rule, an adapter *observes* values your framework already computed
   (gradients from an already-run `.backward()`/`jax.grad`, circuit
   metadata from an already-built tape) -- it never calls your framework's
   differentiation machinery itself.
4. **Auto-populate what you can, defensively.** `CircuitMetadata`/
   `OptimizerMetadata` fields your framework can supply automatically
   (parameter count, optimizer name/learning rate, gradient method) should
   be extracted opportunistically, wrapped so a missing/unexpected
   attribute degrades that one field to `None` rather than raising --
   exactly like `PennyLaneAdapter`'s tape introspection and
   `PyTorchAdapter`'s `_safe()` helper.
5. **Prefer `attach()`/`detach()` for stateful frameworks.** If your
   framework has a persistent object worth holding onto across steps (a
   `torch.nn.Module`, a JAX parameter pytree template, a PennyLane
   `QNode`), give it an `attach(...)`/`detach()`/`attached` lifecycle
   matching the existing adapters, rather than requiring it be re-passed
   on every call.

### Building on `AutogradAdapter` for a new autodiff framework

If your new framework is itself autodiff-based (produces tensors/pytrees
that need `.detach()`/`.cpu()`/`.numpy()`-style conversion, or framework
introspection for a name/learning rate/parameter count), subclass
`qml_observer.adapters.autograd.AutogradAdapter` rather than starting from
scratch -- `PyTorchAdapter`/`JAXAdapter` are both thin subclasses of it and
share this pattern:

```python
from qml_observer.adapters.autograd import AutogradAdapter

class MyFrameworkAdapter(AutogradAdapter):
    framework_name = "myframework"

    def __init__(self, monitor, module=None, **kwargs):
        super().__init__(monitor, **kwargs)
        self._module = module

    def attach(self, module):
        self._module = module
        return self

    def detach(self):
        self._module = None

    @property
    def attached(self):
        return self._module is not None

    def record_step(self, step, loss=None, gradients=None, parameters=None, **kw):
        if gradients is None and self._module is not None:
            gradients = self._collect_gradients_somehow()
        return super().record_step(step, loss=loss, gradients=gradients,
                                    parameters=parameters, **kw)
```

`AutogradAdapter.record_step()` already handles: `to_numpy()` conversion
of `loss`/`gradients`/`parameters`, reducing a scalar-valued `loss` tensor
to a plain `float`, inferring `CircuitMetadata.n_parameters` from
`parameters`'s size when your subclass hasn't already set
`self._n_parameters` some other way, and building `OptimizerMetadata` from
whatever `optimizer_name`/`learning_rate`/`gradient_method` you pass
through.

### Packaging: gate your adapter behind an optional dependency

Every framework-specific adapter lives in its own subpackage
(`qml_observer.adapters.pennylane`, `.qiskit`, `.pytorch`, `.jax`), each
gated behind its own optional dependency, and **none of them are imported
by `qml_observer.adapters.__init__`** -- only `GenericAdapter` and
`AutogradAdapter` (which have zero third-party dependencies) are. Follow
the same pattern for a first-party contribution:

```python
# adapters/myframework/__init__.py
from qml_observer.adapters.myframework.adapter import MyFrameworkAdapter
__all__ = ["MyFrameworkAdapter"]
```

```python
# adapters/myframework/adapter.py
try:
    import myframework
except ImportError as _exc:
    myframework = None
    _IMPORT_ERROR = _exc
else:
    _IMPORT_ERROR = None

def _require_myframework():
    if myframework is None:
        raise ImportError(
            "MyFrameworkAdapter requires the optional 'myframework' dependency. "
            "Install it with `pip install qml-observer[myframework]`."
        ) from _IMPORT_ERROR
```

And add the extra to `pyproject.toml`:

```toml
[project.optional-dependencies]
myframework = ["myframework>=X.Y"]
```

This keeps a plain `pip install qml-observer` free of every optional
framework's dependency footprint, exactly as `torch`/`jax`/`pennylane`/
`qiskit` are today.

## Why write a reporter (or "tracker")?

A reporter is `QMLMonitor`'s output sink -- called once per step
(`record_event`), once with the final diagnosis (`record_diagnosis`), and
once at the end of the run (`finalize`). Write one if you want
`QMLMonitor` output to land somewhere `RunReporter` (JSONL) doesn't reach
by default: an existing experiment tracker (MLflow/W&B, see
[`integrations/experiment_trackers.md`](../integrations/experiment_trackers.md)),
a database, a metrics system, etc.

### The reporter contract (duck type, no ABC)

```python
class MyReporter:
    def record_event(self, event: TrainingEvent) -> None: ...
    def record_diagnosis(self, diagnosis: DiagnosisResult) -> None: ...
    def finalize(self) -> dict: ...
```

Pass an instance as `QMLMonitor(reporter=MyReporter())`; there's nothing
to register or discover beyond that.

### Building on `BaseExperimentTracker` for a tracker-style sink

If your reporter's job is "translate `TrainingEvent`/`DiagnosisResult`
into some third-party service's own metric-logging calls" (the
MLflow/W&B case), subclass
`qml_observer.integrations.trackers.base.BaseExperimentTracker` instead of
implementing the full contract yourself -- it already extracts the
numeric metrics worth logging (`event_metrics()`/`diagnosis_metrics()`)
and wraps every logging call in the project's fail-open policy (addendum
§1: a tracker being unreachable must never propagate into the training
loop). You only implement two hooks:

```python
from qml_observer.integrations.trackers.base import BaseExperimentTracker

class MyTracker(BaseExperimentTracker):
    def __init__(self, client):
        super().__init__()
        self._client = client

    def _log_metrics(self, step: int, metrics: dict) -> None:
        self._client.log(metrics, step=step)

    def _log_summary(self, summary: dict) -> None:
        self._client.set_summary(summary)
```

Gate any third-party dependency the same way adapters do (an
`try`/`except ImportError` guard plus a `_require_...()` helper, and its
own `pyproject.toml` extra) -- see `mlflow_tracker.py`/`wandb_tracker.py`
for the exact pattern.

## Detector plugins (Issue #103)

Third-party detectors are a different extension point from adapters/
reporters above: a detector participates in the diagnosis pipeline
itself (it's driven by `DiagnosisEngine`, not `QMLMonitor.update()`
directly), so it has its own governance and discovery mechanism,
`qml_observer.detectors.plugins` (Milestone 14, Issue #103).

**No registration is required to *use* a detector you've written** --
`QMLMonitor(detectors=[...])`/`DiagnosisEngine(detectors=[...])` already
treat any `BaseDetector` uniformly regardless of where it came from (see
[`development/adding_detectors.md`](adding_detectors.md) for writing one
in the first place). The plugin mechanism here is purely a *discovery*
convenience for detectors installed as separate packages, so consumers
don't need to know your exact import path:

```toml
# your package's own pyproject.toml
[project.entry-points."qml_observer.detectors"]
my_detector = "my_package.detectors:MyDetector"
```

```python
from qml_observer.detectors.plugins import load_detector_plugins
from qml_observer import QMLMonitor

plugin_detectors = load_detector_plugins()   # every installed plugin
monitor = QMLMonitor(detectors=[*builtin_detectors, *plugin_detectors])
```

`list_detector_plugins()` lists what's registered (name -> `module:attr`
target) without importing anything;
`discover_detector_plugins()`/`load_detector_plugins()` import and
validate each one, skipping (with a logged warning) any entry point that
fails to import or doesn't resolve to a `BaseDetector` subclass --
one broken plugin never blocks the rest. `qml-observer plugins list`
exposes the same listing from the CLI.

**No sandboxing.** Per `SECURITY.md`, plugin detectors execute in-process
with full code execution -- the same trust model as installing any other
Python package, not a mitigated or reviewed one. This module does not
attempt to change that; it only makes discovery convenient.

**Proposing a new *built-in* detector** (shipping inside
`qml_observer.detectors` itself, as opposed to your own installable
plugin) goes through a separate RFC process instead --see
[`development/detector_rfc_template.md`](detector_rfc_template.md). That
process does not apply to plugin detectors at all; it exists only for
detectors proposed to join the project's own maintained set.

## Definition of done

Per the blueprint's Volume XVIII, treat a new adapter, reporter, or
detector plugin as complete only once it has: an implementation following
the contracts above, unit tests (and integration tests if it's a full
framework adapter), documentation (a page under `docs/integrations/` for
an adapter, a section of
[`integrations/experiment_trackers.md`](../integrations/experiment_trackers.md)
for a tracker, or your own package's docs for a detector plugin), a
runnable example, defensive/fail-open error handling consistent with
addendum §1, and a `CHANGELOG.md` entry.
