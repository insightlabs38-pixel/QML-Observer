# Architecture overview

QML Observer is a **non-invasive monitoring layer** (plan.md §2): it never
owns the quantum computation itself, only observes training events emitted
by the application and passes them through detection/diagnosis/action
layers. This is what lets it integrate with PennyLane, Qiskit, and fully
custom research code through the same core.

<p align="center">
  <img src="diagrams/overview_architecture.svg" alt="Detailed QML Observer architecture: framework adapters into shared event schemas, QMLMonitor, the statistics engine, detection layer, diagnosis engine, action layer, reporting layer, CLI, and the opt-in telemetry layer" width="820">
</p>

## Layers

1. **Framework adapter layer** (`qml_observer.adapters`) -- converts
   framework-specific objects (a PennyLane `QNode`, a Qiskit
   `QuantumCircuit`/trainer) into the shared event schema. Adapters never
   reimplement a framework's own gradient machinery; they only observe
   already-computed results. See `adapters.md`.

2. **Core event model** (`qml_observer.schemas`) --
   `TrainingEvent`/`GradientSnapshot`/`CircuitMetadata`/`OptimizerMetadata`/
   `DiagnosisResult`: the framework-agnostic data structures every other
   layer is built on. Defined *before* any adapter, per the blueprint's
   Volume II ordering. See `events.md`.

3. **Statistics engine** (`qml_observer.statistics`) -- `RollingWindow`
   (incremental mean/variance/slope over a bounded scalar history) and the
   gradient/loss primitives (`gradient_norm`, `gradient_variance`,
   `loss_slope`, `relative_loss_improvement`, etc.) that detectors are
   built on.

4. **Detection layer** (`qml_observer.detectors`) -- `BaseDetector`
   subclasses that each answer one narrow question ("has the gradient
   collapsed with loss stagnation?", "has the optimizer frozen?", "has the
   loss converged to a good value?") and return a `DetectorResult`, never a
   final verdict. See `detectors.md`.

5. **Diagnosis engine** (`qml_observer.diagnosis`) -- combines every
   detector's `DetectorResult` into one explainable `DiagnosisResult` via
   weighted evidence (`diagnosis/scoring.py`), with `CONVERGED` given
   explicit priority over any other simultaneously-triggered issue. This
   layer is deliberately deterministic and rule-based, not ML-based, for
   the MVP -- see the blueprint's Volume VII.

6. **Action layer** (`qml_observer.actions`) -- `ActionPolicy` selects
   `LogAction`/`AlertAction`/`StopAction` per diagnosis based on the
   configured mode (`"log"`/`"warn"`/`"pause"`/`"stop"`/`"adaptive"`).
   Conservative by default; never escalates a `degraded` diagnosis to a
   stop except under explicit opt-in. See `actions.md`.

7. **Reporting layer** (`qml_observer.reporting`) -- `RunReporter` +
   JSONL event/diagnosis/summary logging, `build_run_summary()`, and
   `estimate_compute_saved()`. Entirely framework-agnostic: it consumes
   whatever `QMLMonitor` produces regardless of which adapter drove it,
   which is why the same reporting/CLI/benchmark infrastructure applies
   identically to PennyLane and Qiskit runs without any adapter-specific
   code.

8. **CLI** (`qml_observer.cli`) -- `qml-observer inspect`/`report` reading
   JSONL logs produced by the reporting layer, plus
   `qml-observer telemetry {enable,disable,status}` for the layer below.

9. **Telemetry** (`qml_observer.telemetry`) -- optional, **disabled by
   default** anonymized calibration telemetry (addendum §5). `consent.py`
   persists an explicit opt-in decision (never assumed, never enabled in
   a non-interactive environment); `TelemetryCollector` builds an
   anonymized `TelemetryRecord` (detector names, extracted numeric
   thresholds, diagnosis issue/confidence, a coarse qubit-count bucket,
   detection latency -- never raw gradients, loss, circuit structure,
   parameters, run IDs, file paths, or hostnames) at `QMLMonitor.finish()`
   and either queues it locally as JSON Lines or POSTs it to an
   explicitly configured endpoint. Wholly independent of the
   detection/diagnosis/action path above: a broken or misconfigured
   collector can never affect a diagnosis or the training loop (same
   fail-open guarantee as §`QMLMonitor`: the seam, below). See
   `../development/telemetry.md` for the published schema.

## `QMLMonitor`: the seam

`qml_observer.core.monitor.QMLMonitor` is the single public object every
layer above flows through: adapters call `monitor.update(...)`, which
maintains rolling `RunState`, runs the configured detectors through
`DiagnosisEngine.evaluate()`, runs the configured `ActionPolicy`, and
optionally streams every event/diagnosis to a `reporter`. This is also
where the fail-open guarantee lives (addendum §1): any exception raised
inside `update()`/`finish()` -- schema validation, gradient summarization,
detector/statistics/action-layer calls -- is caught, logged with a full
traceback, and converted into a `degraded=True` diagnosis rather than
propagated into the caller's training loop.

## Why this separation matters

The blueprint states two rules as more important than any individual
feature:

1. **Keep the detection engine independent from the quantum framework.**
   PennyLane and Qiskit are responsible for producing training
   observations; QML Observer is responsible for interpreting them. This
   is why `PennyLaneAdapter`/`QiskitAdapter` are thin translation layers,
   not where any detection logic lives.
2. **Separate detection from diagnosis.** A detector reports raw evidence
   ("gradient collapse detected"); the diagnosis engine decides what that
   evidence is more consistent with. Conflating the two would make it
   impossible to reason about, test, or extend detectors independently of
   how their evidence gets combined.

## Concurrency

`QMLMonitor` and `RollingWindow` are **not thread-safe by default** in
`0.x` (addendum §8) -- `update()` mutates `RunState`'s rolling windows
in place with no locking, so concurrent calls from multiple threads on
the same monitor instance can race. For multi-process/distributed
training (e.g. PyTorch DDP-style hybrid workflows), the recommended
`0.x` pattern is **one `QMLMonitor` per process/rank**, with any
cross-run aggregation done post-hoc in the reporting layer rather than
live cross-process merging. True distributed-aware monitoring (shared,
synchronized state across ranks) is a post-1.0 feature tracked under
Milestone 14.
