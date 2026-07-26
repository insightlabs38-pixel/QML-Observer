# Installation

QML Observer is still pre-1.0 (`Development Status :: 2 - Pre-Alpha`) and
is not yet published to PyPI. Install from a clone of the repository:

```bash
git clone https://github.com/qml-observer/qml-observer
cd qml-observer
pip install -e .
```

## Requirements

- Python 3.12 or 3.13 (rolling two-version support window -- see
  `CONTRIBUTING.md`).
- `numpy>=1.26` (the only hard dependency of the core package).

## Optional framework integrations

The core package (event schemas, statistics, detectors, diagnosis, actions,
reporting, CLI) has **no** PennyLane or Qiskit dependency -- adapters are
opt-in extras:

```bash
pip install -e ".[pennylane]"   # PennyLane adapter
pip install -e ".[qiskit]"      # Qiskit adapter (qiskit + qiskit-machine-learning)
pip install -e ".[dev]"         # pytest, ruff, mypy, pre-commit -- for contributors
```

Multiple extras can be combined: `pip install -e ".[pennylane,qiskit,dev]"`.

If you import `qml_observer.adapters.pennylane` or
`qml_observer.adapters.qiskit` without the corresponding extra installed,
you'll get a clear `ImportError` naming the missing package rather than a
confusing failure deep in adapter internals.

## Verifying the install

```bash
python -c "import qml_observer; print(qml_observer.__version__)"
```

If that prints a version string (e.g. `0.1.0`), the install worked. Next:
[`quickstart.md`](quickstart.md).
