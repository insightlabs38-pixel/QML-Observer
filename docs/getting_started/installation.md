# Installation

QML Observer is in public beta (`0.6.0`, `Development Status :: 4 - Beta`)
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
reporting, telemetry, CLI) has **no** PennyLane, Qiskit, PyTorch, JAX,
MLflow, or W&B dependency -- adapters and tracker integrations are all
opt-in extras, installable directly from PyPI:

```bash
pip install "qml-observer[pennylane]"   # PennyLane adapter
pip install "qml-observer[qiskit]"      # Qiskit adapter (qiskit + qiskit-machine-learning)
pip install "qml-observer[torch]"       # PyTorch adapter (Milestone 14)
pip install "qml-observer[jax]"         # JAX adapter (Milestone 14)
pip install "qml-observer[mlflow]"      # MLflow experiment-tracker integration (Milestone 14)
pip install "qml-observer[wandb]"       # Weights & Biases experiment-tracker integration (Milestone 14)
```

From a local checkout, use the editable-install form instead:

```bash
pip install -e ".[pennylane]"
pip install -e ".[qiskit]"
pip install -e ".[torch]"
pip install -e ".[jax]"
pip install -e ".[mlflow]"
pip install -e ".[wandb]"
pip install -e ".[dev]"          # pytest, ruff, mypy, pre-commit -- for contributors
```

Multiple extras can be combined:
`pip install "qml-observer[pennylane,qiskit,torch,jax]"` (or
`pip install -e ".[pennylane,qiskit,torch,jax,mlflow,wandb,dev]"` from a
checkout).

If you import `qml_observer.adapters.pennylane`,
`qml_observer.adapters.qiskit`, `qml_observer.adapters.pytorch`,
`qml_observer.adapters.jax`, or the tracker modules under
`qml_observer.integrations.trackers` without the corresponding extra
installed, you'll get a clear `ImportError` naming the missing package
rather than a confusing failure deep in adapter internals.
`qml_observer.adapters.autograd.AutogradAdapter` (the shared,
dependency-free base `PyTorchAdapter`/`JAXAdapter` build on) requires no
extra at all.

## Optional dashboard (Milestone 11)

A read-only web dashboard (loss/gradient charts, diagnosis panel,
compute-usage panel -- see `docs/architecture/dashboard.md`) ships as a
separate optional extra, since most users won't need it for scripted/CI
runs:

```bash
pip install "qml-observer[dashboard]"     # from PyPI
pip install -e ".[dashboard]"             # from a checkout
```

Importing `qml_observer.dashboard` itself never requires this extra --
only actually creating/serving the app (`create_app`/`run_dashboard`)
does, and raises a clear `ImportError` with install instructions if it's
missing.


## Verifying the install

```bash
python -c "import qml_observer; print(qml_observer.__version__)"
qml-observer --help
```

The first command should print a version string (e.g. `0.6.0`); the
second should list the CLI's subcommands (`inspect`, `report`, `run`,
`benchmark`, `telemetry`). If both work, the install succeeded. Next:
[`quickstart.md`](quickstart.md).

## Known limitations of this release

See the project [README](https://github.com/insightlabs38-pixel/QML-Observer#known-limitations) for the current, authoritative list (diagnoses are probabilistic rather
than proof, `"pause"` currently behaves as `"warn"`, simulator-only,
etc.) -- kept in one place to avoid the two copies drifting out of sync.
