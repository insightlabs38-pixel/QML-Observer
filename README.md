# QML Observer

An open-source observability and diagnostic framework for variational quantum
machine learning (QML) training. QML Observer watches training runs in real
time, detects pathologies such as probable barren plateaus, stagnation, and
noise-dominated optimization, and can log, warn, pause, or stop training
before expensive quantum computation is wasted.

> **Status:** early development (Milestone 0/1). Core schemas and monitoring
> engine are being built incrementally — see the issue tracker for progress.

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

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.12 or 3.13.

## License

Mozilla Public License 2.0 (MPL-2.0). See [`LICENSE`](./LICENSE) and
[`CONTRIBUTING.md`](./CONTRIBUTING.md#license) for rationale.
