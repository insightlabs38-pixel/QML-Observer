# QML Observer

An open-source observability and diagnostic framework for variational quantum
machine learning (QML) training. QML Observer watches training runs in real
time, detects pathologies such as probable barren plateaus, stagnation, and
noise-dominated optimization, and can log, warn, pause, or stop training
before expensive quantum computation is wasted.

> **Status:** v0.1.0 — first public MVP release. Core schemas, monitoring
> engine, detectors, diagnosis engine, actions, both the PennyLane and
> Qiskit adapters, JSONL logging, run summaries, compute-saved estimation,
> the CLI, and the calibration benchmark suite are all shipped (Milestones
> 0–8). See `CHANGELOG.md` for the full release notes and
> `docs/roadmap.md` for what's next.

## Scope note

QML Observer targets **simulator-based training only** through the MVP and
all `0.x` releases. Real hardware / cloud QPU integration (IBM Quantum, AWS
Braket, IonQ, etc.) is explicitly out of scope until dedicated funding is
secured. See `docs/roadmap.md` for what hardware support would require.

## Positioning

QML Observer identifies **probable plateau-like failure modes** based on
observed training signals — it does not claim definitive, guaranteed barren
plateau detection in all settings. This distinction is central to the
project's scientific credibility.

## Examples

Runnable examples live under `examples/`. PennyLane examples (requires
`pip install -e ".[dev,pennylane]"`):

- `examples/pennylane/basic_monitor.py` — minimal QNode + `QMLMonitor`
  integration, no detectors.
- `examples/pennylane/barren_plateau_demo.py` — the project's flagship
  demo: a healthy run that is never stopped early, contrasted with a
  collapsed-gradient run that is stopped early with an estimated
  compute-saved figure.
- `examples/pennylane/noisy_training.py` — training under a small finite
  shot budget, showing noisier per-step gradient statistics without
  false-positive plateau detection.

Qiskit examples (requires `pip install -e ".[dev,qiskit]"`):

- `examples/qiskit/basic_monitor.py` — minimal `QuantumCircuit` +
  `QMLMonitor` integration via manual parameter-shift gradients, no
  detectors.
- `examples/qiskit/barren_plateau_demo.py` — the Qiskit version of the
  flagship demo: a healthy run that is never stopped early, contrasted
  with a collapsed-gradient run that is stopped early with an estimated
  compute-saved figure.
- `examples/qiskit/vqc_callback_demo.py` — wires `QiskitAdapter.callback`
  directly into a real `qiskit-machine-learning` `VQC` trainer's own
  `callback=` hook, with no manual training loop at all.

## Reporting & CLI

Wire a `RunReporter` into `QMLMonitor` to get a JSONL event/diagnosis log
plus a run summary (including an estimated compute-saved figure) on
`finish()`:

```python
from qml_observer import QMLMonitor
from qml_observer.reporting.reporter import RunReporter

reporter = RunReporter("runs/run.jsonl", framework="pennylane", planned_steps=1000)
monitor = QMLMonitor(reporter=reporter, planned_steps=1000)
```

Then inspect the log from the command line:

```bash
qml-observer report runs/run.jsonl   # human-readable run summary
qml-observer inspect runs/run.jsonl  # every logged record as pretty JSON
```

See [`docs/integrations/qiskit.md`](./docs/integrations/qiskit.md) for the
Qiskit adapter guide.

## Benchmarks & calibration

`benchmarks/run_benchmarks.py` reproduces the false-positive-rate and
detection-latency numbers that chose the detectors' default thresholds;
`benchmarks/qml_observer_benchmark.ipynb` walks through the same results
alongside the live "critical MVP demo". See
[`docs/research/validation.md`](./docs/research/validation.md) for the
full methodology and current results.

## Documentation

Full docs (installation, quickstart, "how to interpret alerts",
architecture, per-detector guides, calibration methodology, and
contributor guides) live under [`docs/`](./docs/index.md).

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.12 or 3.13.

## License

Mozilla Public License 2.0 (MPL-2.0). See [`LICENSE`](./LICENSE) and
[`CONTRIBUTING.md`](./CONTRIBUTING.md#license) for rationale.
