# Contributing to QML Observer

Thanks for your interest in contributing. This document will grow alongside
`docs/development/contributing.md` as the project matures.

## Development setup

```bash
git clone <repo>
cd qml-observer
pip install -e ".[dev]"
pre-commit install
```

Run the test suite:

```bash
pytest
```

Lint and type-check:

```bash
ruff check .
mypy src
```

## Python version support policy

QML Observer supports a **rolling two-version window** of upstream Python
releases. Currently supported: **3.12 and 3.13**. As new stable Python
minors ship, the oldest supported minor is dropped. Check `pyproject.toml`
(`requires-python`) and `.github/workflows/ci.yml` for the exact matrix
in effect at any given time.

## Why MPL-2.0?

QML Observer is licensed under the Mozilla Public License 2.0 (file-level
weak copyleft), not Apache-2.0 or MIT, for one deliberate reason: it
prevents the core detection/diagnosis engine from being silently forked
into closed-source commercial products, while still permitting the project
to be linked into larger proprietary training pipelines — a likely use case
for teams building on top of cloud/QPU workflows. Modifications to
qml-observer's own files must be shared back; code that merely *uses*
qml-observer as a library does not inherit copyleft obligations.

We considered Apache-2.0 (matching PennyLane and Qiskit, with an explicit
patent grant) as a more frictionless, non-copyleft alternative, and may
revisit this tradeoff if it becomes a real adoption barrier. Any future
license change will be called out prominently in `CHANGELOG.md`.

## Detector proposals

New built-in detectors should go through the RFC template in
`docs/development/detector_rfc_template.md` (added alongside Milestone 14).
A third-party detector plugin API is planned for Milestone 14 (not yet
shipped as of `0.3.0`); when it lands, plugin detectors will run in-process
with no sandboxing, and that security boundary will be documented
explicitly rather than silently assumed. See
`docs/development/data_handling.md` for the current data-retention and
privacy model.
