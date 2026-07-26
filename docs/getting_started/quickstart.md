# Quickstart

This walks through the smallest useful integration: a manual training loop,
the three MVP detectors, and a `"warn"` policy. Every snippet on this page
is copy-pasteable and has been run against the actual package.

## 1. Wrap your loop

```python
import numpy as np
from qml_observer import QMLMonitor
from qml_observer.adapters.generic import GenericAdapter
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.stagnation import StagnationDetector
from qml_observer.detectors.convergence import ConvergenceDetector

monitor = QMLMonitor(
    detectors=[BarrenPlateauDetector(), StagnationDetector(), ConvergenceDetector()],
    policy="warn",  # "log" | "warn" | "pause" | "stop" | "adaptive"
)
adapter = GenericAdapter(monitor)

rng = np.random.default_rng(0)
loss = 1.0
for step in range(30):
    loss *= 0.9  # replace with your real training step
    gradients = rng.normal(0, 0.1, size=5)  # replace with your real gradients
    diagnosis = adapter.record(step, loss=loss, gradients=gradients)

    if monitor.should_stop():
        print("Monitor requested a stop:", diagnosis.issue.value)
        break

print(diagnosis.issue.value, diagnosis.confidence)
```

With `policy="warn"`, a plateau-like diagnosis prints a terminal warning but
never halts your loop -- you decide what to do with `monitor.should_stop()`.
Use `policy="stop"` to have `should_stop()` return `True` once a
detector's evidence clears its threshold (see
[`concepts.md`](concepts.md) for what each policy mode actually does).

## 2. Read the diagnosis

Every call to `adapter.record(...)` (or `monitor.update(...)` directly)
returns a `DiagnosisResult`:

```python
diagnosis.issue  # IssueType, e.g. IssueType.POSSIBLE_BARREN_PLATEAU
diagnosis.confidence  # float in [0, 1]
diagnosis.severity  # "info" | "warning" | "critical"
diagnosis.evidence  # list[str], human-readable
diagnosis.recommendations  # list[str], human-readable
diagnosis.degraded  # True if a detector/statistics call failed mid-run
```

See [`getting_started/concepts.md#how-to-interpret-alerts`](concepts.md#how-to-interpret-alerts)
for what each `issue` value means and how much to trust it.

## 3. Use a real framework instead of `GenericAdapter`

Swap `GenericAdapter` for `PennyLaneAdapter` or `QiskitAdapter` and pass a
`QNode`/`QuantumCircuit` instead of raw floats/arrays:

```python
from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter

adapter = PennyLaneAdapter(monitor)
adapter.attach(my_qnode)
# ... adapter.record_step(step, loss, gradients, parameters=params)
```

See [`integrations/pennylane.md`](../integrations/pennylane.md) and
[`integrations/qiskit.md`](../integrations/qiskit.md) for the full guides,
including finite-shots handling, gradient-method metadata, and circuit
metadata extraction.

## 4. Log to disk and get a run report

```python
from qml_observer.reporting.reporter import RunReporter

monitor = QMLMonitor(
    detectors=[BarrenPlateauDetector(), StagnationDetector(), ConvergenceDetector()],
    policy="stop",
    reporter=RunReporter(jsonl_path="run.jsonl"),
    planned_steps=10_000,  # enables "estimated compute saved" in the final report
)
```

Then, from the command line:

```bash
qml-observer report run.jsonl
```

prints the blueprint's Volume XV-style human-readable summary -- status,
evidence, confidence, and estimated compute saved. `qml-observer inspect
run.jsonl` dumps every raw JSONL record instead, useful for debugging.

`RunReporter`'s automatic summary (above) only ever sees the bare
`TrainingEvent` `QMLMonitor` forwards to it, so it never includes
gradient/circuit/optimizer detail. For the fuller report (matching the
CLI's `Gradient norm:`/`Gradient variance:` lines), call
`build_run_summary(monitor.state, final_diagnosis)` directly after
`finish()` and write it as the `"summary"` JSONL record yourself:

```python
from qml_observer.reporting.jsonl import JSONLWriter, summary_record
from qml_observer.reporting.summary import build_run_summary

final = monitor.finish()
summary = build_run_summary(monitor.state, final, framework="pennylane")
with JSONLWriter("run.jsonl") as writer:
    writer.write(summary_record(summary))
```

## 5. See the full demo

`examples/pennylane/barren_plateau_demo.py` and
`examples/qiskit/barren_plateau_demo.py` run the blueprint's "critical MVP
demo" end to end: a healthy run that completes normally, contrasted with an
engineered collapsed-gradient run that is stopped early with an estimated
compute-saved figure. `benchmarks/qml_observer_benchmark.ipynb` walks
through the same demo alongside the calibration benchmarks that chose the
detectors' default thresholds.
