# Installation

QML Observer is in public beta (`0.2.0`, `Development Status :: 4 - Beta`)
and is published on PyPI:

```bash
pip install qml-observer
```

For local development or to work from a checkout of the repository:

```bash
git clone https://github.com/insightlabs38-pixel/QML-Observer
cd QML-Observer
pip install -e .
```

## Requirements

- Python 3.12 or 3.13 (rolling two-version support window -- see
  `CONTRIBUTING.md`).
- `numpy>=1.26` (the only hard dependency of the core package).

## Optional framework integrations

The core package (event schemas, statistics, detectors, diagnosis, actions,
reporting, telemetry, CLI) has **no** PennyLane or Qiskit dependency --
adapters are opt-in extras, installable directly from PyPI:

```bash
pip install "qml-observer[pennylane]"   # PennyLane adapter
pip install "qml-observer[qiskit]"      # Qiskit adapter (qiskit + qiskit-machine-learning)
```

From a local checkout, use the editable-install form instead:

```bash
pip install -e ".[pennylane]"
pip install -e ".[qiskit]"
pip install -e ".[dev]"          # pytest, ruff, mypy, pre-commit -- for contributors
```

Multiple extras can be combined:
`pip install "qml-observer[pennylane,qiskit]"` (or
`pip install -e ".[pennylane,qiskit,dev]"` from a checkout).

If you import `qml_observer.adapters.pennylane` or
`qml_observer.adapters.qiskit` without the corresponding extra installed,
you'll get a clear `ImportError` naming the missing package rather than a
confusing failure deep in adapter internals.

## Verifying the install

```bash
python -c "import qml_observer; print(qml_observer.__version__)"
qml-observer --help
```

The first command should print a version string (e.g. `0.2.0`); the
second should list the CLI's subcommands (`inspect`, `report`, `run`,
`benchmark`, `telemetry`). If both work, the install succeeded. Next:
[`quickstart.md`](quickstart.md).

## Known limitations of this release

See the project [README](https://github.com/insightlabs38-pixel/QML-Observer#known-limitations) for the current, authoritative list (diagnoses are probabilistic rather
than proof, `"pause"` currently behaves as `"warn"`, simulator-only,
etc.) -- kept in one place to avoid the two copies drifting out of sync.
