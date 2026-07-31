# QML Observer

[![CI](https://github.com/insightlabs38-pixel/QML-Observer/actions/workflows/ci.yml/badge.svg)](https://github.com/insightlabs38-pixel/QML-Observer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qml-observer.svg?cacheSeconds=3600)](https://pypi.org/project/qml-observer/)
[![Python versions](https://img.shields.io/pypi/pyversions/qml-observer.svg)](https://pypi.org/project/qml-observer/)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-brightgreen.svg)](./LICENSE)

An open-source observability and diagnostic framework for variational quantum
machine learning (QML) training. QML Observer watches training runs in real
time, detects pathologies such as probable barren plateaus, stagnation, and
noise-dominated optimization, and can log, warn, pause, or stop training
before expensive quantum computation is wasted.

> **Status:** v0.5.0 — public beta. Core schemas, monitoring engine,
> detectors, diagnosis engine, actions, both the PennyLane and Qiskit
> adapters, JSONL logging, run summaries, compute-saved estimation, the
> CLI, the calibration benchmark suite, webhook alerting (including a
> Slack-compatible formatter, alert deduplication/cooldowns, evidence
> redaction, and a webhook-URL SSRF safeguard), an optional read-only
> dashboard (`qml-observer[dashboard]`: live loss/gradient charts, a
> diagnosis panel, compute-usage panel, run history, and data export),
> and opt-in research-grade diagnostics (`qml_observer.advanced`: QFIM
> estimation/conditioning, parameter-redundancy detection,
> Hessian-vector products, loss-landscape sampling, and qubit/depth
> gradient-variance scaling analysis — see `docs/research/geometry.md`)
> are all shipped (Milestones 0–12). See `CHANGELOG.md` for the full
> release notes and `docs/roadmap.md` for what's next. **The `0.x` API is
> not yet stable and may change without a major-version bump**, per
> SemVer's `0.x` convention.
>
> Note: the `"pause"` action-policy mode currently behaves identically to
> `"warn"` — a distinct pause-and-preserve-state action (`PauseAction`)
> is planned for Milestone 13 and is not yet implemented. See
> `docs/architecture/actions.md`.

## Architecture

<p align="center">
  <img src="docs/architecture/diagrams/readme_architecture.svg" alt="QML Observer pipeline: training loop through an adapter into QMLMonitor, statistics, detectors, diagnosis engine, and action policy, which logs, warns, or stops training" width="260">
</p>

Training events flow one-way through the pipeline above; nothing here ever
owns or drives the quantum computation itself (plan.md §2). See
[`docs/architecture/overview.md`](docs/architecture/overview.md) for the
full, module-by-module breakdown, including the diagnosis engine's
weighted-evidence scoring and the opt-in telemetry layer.
