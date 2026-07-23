# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) once
it reaches `0.1.0`.

## [Unreleased]

### Added
- Project skeleton: `pyproject.toml`, `src/` layout, `LICENSE` (MPL-2.0),
  `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Milestone 0, Issue #1).
- `qml_observer.schemas.training.TrainingEvent` (Milestone 1, Issue #4).
- `qml_observer.core.monitor.QMLMonitor` core lifecycle, rolling run
  state, and generic adapter (Milestone 2, Issues #11-#17).
- `qml_observer.statistics.gradients`: `gradient_norm`, `gradient_variance`,
  `gradient_percentiles`, `mean_absolute_gradient` (Milestone 3, Issues
  #18-#20).
- `qml_observer.statistics.loss`: `loss_slope`, `relative_loss_improvement`,
  `is_loss_stagnant` (Milestone 3, Issues #21-#22).
- `qml_observer.statistics.rolling.RollingWindow`: incremental
  mean/variance/slope over a bounded scalar history (Milestone 3, Issue #23).
- Consolidated numerical stability test suite covering addendum §7 edge
  cases: empty gradient arrays, NaN/Inf propagation, and zero/one-observation
  `RollingWindow` behavior (Milestone 3, Issue #24).
