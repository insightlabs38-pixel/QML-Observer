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
- `qml_observer.detectors.base.BaseDetector` / `DetectorResult`: shared
  detector interface (Milestone 4, Issue #25).
- `qml_observer.detectors.barren_plateau.BarrenPlateauDetector`: sustained
  gradient collapse + loss stagnation detection; never triggers on a small
  gradient alone (Milestone 4, Issue #26).
- `qml_observer.detectors.stagnation.StagnationDetector`: frozen-optimizer
  detection via loss/parameter/learning-rate signals (Milestone 4, Issue #27).
- `qml_observer.detectors.convergence.ConvergenceDetector`: distinguishes
  good convergence (low absolute loss) from bad gradient collapse
  (Milestone 4, Issue #28).
- `qml_observer.diagnosis.engine.DiagnosisEngine`: combines detector output
  into a single, explainable `DiagnosisResult`, with `CONVERGED` given
  explicit priority over other simultaneously-triggered issues; wired into
  `QMLMonitor._evaluate()` (Milestone 4, Issue #29).
- `qml_observer.diagnosis.scoring.combine_detector_results`: standalone,
  reusable confidence-combination primitive extracted out of
  `DiagnosisEngine`, with weighted noisy-OR combination across detectors
  that agree on the same issue (Milestone 4, Issue #30).
- `qml_observer.diagnosis.explanations.explain`: renders a `DiagnosisResult`
  as a human-readable, multi-line explanation (headline, confidence,
  degraded-mode banner, evidence, recommendations) for CLI/report/alert
  consumers (Milestone 4, Issue #31).
- `tests/fixtures/synthetic_runs.py`: five seeded synthetic training-run
  generators (healthy learning, convergence, artificial plateau,
  noise-dominated, stagnant optimizer) covering plan.md §15's benchmark
  categories, plus validation tests confirming each detector/diagnosis
  combination classifies every scenario correctly and does not
  false-positive on noise alone (Milestone 4, Issue #32).
- `qml_observer.actions.base.Action` / `ActionResult`: shared action
  interface, mirroring `detectors.base.BaseDetector` (Milestone 5,
  Issue #33).
- `qml_observer.actions.log.LogAction`: always-executes, never-raises
  diagnosis logging -- intervention level 1 (Milestone 5, Issue #34).
- `qml_observer.actions.alert.AlertAction`: terminal + logger warning for
  non-`"info"`-severity diagnoses -- intervention level 2 (Milestone 5,
  Issue #35).
- `qml_observer.actions.stop.StopAction`: records a stop request via a
  `.triggered` flag for the caller's training loop to check; never
  reaches into the loop directly, per the non-invasive core principle
  (Milestone 5, Issue #36).
- `qml_observer.actions.policies.ActionPolicy`: selects `log`/`alert`/
  `stop` per diagnosis for the `"log"`, `"warn"`, `"pause"`, `"stop"`,
  and `"adaptive"` modes; enforces addendum §1's conservative
  degraded-diagnosis rule (a `degraded=True` result never escalates to
  `StopAction` unless `mode="adaptive"` **and** the explicit
  `allow_stop_on_degraded=True` flag is set) (Milestone 5, Issues
  #37-#39). `"pause"` behaves as `"warn"` until `PauseAction` ships.
- `QMLMonitor` is now wired to a real `ActionPolicy` instead of its
  previous placeholder `should_stop()` logic: `update()`/`finish()` run
  the policy each step (exposed via `latest_action_result()`), and
  `should_stop()` is a pure recomputation from `state.latest_diagnosis`
  via `ActionPolicy.select_action()`. A new `action_policy` constructor
  argument allows advanced configuration (e.g. `mode="adaptive"` with
  `allow_stop_on_degraded=True`, or custom `Action` injection for
  testing). Action-layer failures are caught the same fail-open way as
  detector/statistics failures, so a broken custom `Action` cannot crash
  a training loop (Milestone 5, Issues #38-#40).
- `qml_observer.adapters.pennylane.adapter.PennyLaneAdapter`: first real
  framework integration. `attach()`/`detach()` a PennyLane `QNode`;
  `record_step()` observes already-computed `loss`/`gradients`/
  `parameters` (never reimplements PennyLane's gradient machinery) and
  auto-populates `CircuitMetadata` and `OptimizerMetadata` on every call
  (Milestone 6, Issue #41). Requires the optional `pennylane` extra
  (`pip install qml-observer[pennylane]`); raises a clear `ImportError`
  otherwise.
- `PennyLaneAdapter` records the QNode's configured `diff_method` into
  `OptimizerMetadata.gradient_method`, verified for both
  `"parameter-shift"` (Milestone 6, Issue #42) and `"adjoint"`
  (Milestone 6, Issue #43).
- `PennyLaneAdapter` infers the shot count for finite-shots devices from
  the constructed tape (falling back to the device default), reporting
  `None` for analytic (`shots=None`) circuits; an explicit `shots=`
  argument to `record_step()` always overrides inference
  (Milestone 6, Issue #44).
- `PennyLaneAdapter.extract_circuit_metadata()`: builds `CircuitMetadata`
  (qubits, depth, gate count, entangling-gate count, parameter count)
  from a PennyLane tape/`QuantumScript`, extracting every field
  defensively so an unexpected PennyLane version/tape shape degrades to
  `None` fields instead of raising (Milestone 6, Issue #45).

