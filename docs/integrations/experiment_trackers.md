# Experiment-tracker integrations (MLflow, Weights & Biases)

**Requires:** `pip install qml-observer[mlflow]` and/or
`pip install qml-observer[wandb]`

`MLflowTracker` and `WandbTracker`
(`qml_observer.integrations.trackers.mlflow_tracker`/`.wandb_tracker`)
write `QMLMonitor` output into *existing* experiment-tracking
infrastructure, rather than adding a competing tracker of qml_observer's
own. Both implement the same `RunReporter` duck type
`QMLMonitor(reporter=...)` already knows how to drive
(`record_event`/`record_diagnosis`/`finalize` -- see
[`architecture/events.md`](../architecture/events.md) and
`reporting/reporter.py`), so either can be passed directly in place of
`RunReporter`.

Neither tracker starts or ends a run itself -- attach to a run you've
already opened yourself (`mlflow.start_run()` / `wandb.init()`), matching
this milestone's framing of "writing output into existing tracking infra,
not a competing tracker."

## MLflow

```python
import mlflow
from qml_observer import QMLMonitor
from qml_observer.integrations.trackers.mlflow_tracker import MLflowTracker

with mlflow.start_run():
    monitor = QMLMonitor(reporter=MLflowTracker())
    for step in range(1000):
        monitor.update(step=step, loss=loss)
    monitor.finish()
```

Logging from a process without an "active" MLflow run (e.g. a worker that
didn't call `mlflow.start_run()` itself)? Pass an explicit `run_id`
(logged to via `mlflow.tracking.MlflowClient` instead of the module-level
fluent API) and, if needed, `tracking_uri`:

```python
tracker = MLflowTracker(run_id=run_id, tracking_uri="http://mlflow.internal:5000")
monitor = QMLMonitor(reporter=tracker)
```

Per-step `loss`/`wall_time` are logged as MLflow metrics (keyed by
`step`); the final `DiagnosisResult` (issue, confidence, severity,
degraded flag) is logged as `qml_observer.*`-prefixed tags at
`finalize()` time.

## Weights & Biases

```python
import wandb
from qml_observer import QMLMonitor
from qml_observer.integrations.trackers.wandb_tracker import WandbTracker

run = wandb.init(project="qml-experiments")
monitor = QMLMonitor(reporter=WandbTracker(run=run))
for step in range(1000):
    monitor.update(step=step, loss=loss)
monitor.finish()
run.finish()
```

If `run=` is omitted, `WandbTracker` resolves `wandb.run` (the currently
active run, if any) lazily at logging time -- so it still works even if
`wandb.init()` hadn't been called yet when the tracker itself was
constructed. Per-step metrics are logged via `run.log(...)`; the final
diagnosis is written into `run.summary` at `finalize()` time.

## Fail-open behavior

Both trackers wrap every logging call in the same fail-open policy
`QMLMonitor` itself follows (addendum §1): a tracker being unreachable
(no active run, network issue, disabled run, missing credentials, etc.)
is logged at `warning` level and never propagates into your training
loop.

## Using both a tracker and `RunReporter`/JSONL logging

Neither tracker replaces `RunReporter` (Milestone 7's JSONL logging) --
they're an additional sink. `QMLMonitor` accepts one `reporter=`, so to
use both, fan out from a small wrapper:

```python
class FanOutReporter:
    def __init__(self, *reporters):
        self._reporters = reporters

    def record_event(self, event):
        for r in self._reporters:
            r.record_event(event)

    def record_diagnosis(self, diagnosis):
        for r in self._reporters:
            r.record_diagnosis(diagnosis)

    def finalize(self):
        return [r.finalize() for r in self._reporters][0]


reporter = FanOutReporter(RunReporter("run.jsonl"), MLflowTracker())
monitor = QMLMonitor(reporter=reporter)
```

## Writing your own tracker integration

Both trackers subclass
`qml_observer.integrations.trackers.base.BaseExperimentTracker`, which
factors out the `record_event`/`record_diagnosis`/`finalize` skeleton and
the fail-open error handling -- a third-party tracker only needs to
implement `_log_metrics(step, metrics)` and `_log_summary(summary)`. See
[`development/plugin_api.md`](../development/plugin_api.md) for the full
plugin-authoring guide.
