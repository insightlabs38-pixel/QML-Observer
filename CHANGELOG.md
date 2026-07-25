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
- `examples/pennylane/`: `basic_monitor.py` (minimal integration, no
  detectors), `barren_plateau_demo.py` (the blueprint's Volume XX "critical
  MVP demo" -- a healthy run that is never stopped early, contrasted with an
  engineered collapsed-gradient run that is stopped early with an estimated
  compute-saved figure), and `noisy_training.py` (finite-shots training
  showing noisier gradient statistics without false-positive plateau
  detection) (Milestone 6, Issue #46).
- `tests/integration/pennylane/`: end-to-end tests driving a real `QNode`,
  a real PennyLane optimizer, and real `qml.grad()` gradients through the
  full adapter -> monitor -> detector -> diagnosis -> action pipeline,
  covering healthy convergence, the engineered plateau scenario, both
  parameter-shift and adjoint differentiation, finite shots, and fail-open
  behavior with a real QNode (Milestone 6, Issue #47).
- `qml_observer.adapters.qiskit.adapter.QiskitAdapter`: `attach()`/`detach()`
  lifecycle accepting a `QuantumCircuit` directly or a trainer object
  exposing one (`.circuit`/`.ansatz`, e.g. `VQC`/`NeuralNetworkClassifier`),
  and `record_step()`/`record_gradient()` observing loss/gradients/parameters,
  never reimplementing Qiskit's own optimization or gradient machinery
  (Milestone 8, Issue #58).
- `QiskitAdapter.callback()`: a single callback entry point normalizing
  across the Qiskit optimizer/trainer callback shapes seen in practice --
  `qiskit-machine-learning` trainer style (`weights, obj_func_eval`),
  `qiskit_algorithms`/`qiskit-machine-learning` `SPSA`-style (`nfev, params,
  fval, stepsize, accepted`), plain `scipy.optimize.minimize`-style (`xk`
  only), and the blueprint's own manual `(iteration, parameters, loss)`
  form -- so it can be passed directly as `callback=adapter.callback`
  (Milestone 8, Issue #59).
- `QiskitAdapter.normalize_optimizer_metadata()`: best-effort
  `OptimizerMetadata` extraction from a live Qiskit optimizer object via
  its `.settings` dict, handling inconsistent learning-rate keys across
  optimizer classes (`"learning_rate"` for `SPSA`, `"lr"` for `ADAM`, none
  for gradient-free optimizers like `COBYLA`) and inferring
  `gradient_method` (e.g. `"spsa-approximation"`, `"gradient-free"`) from
  known optimizer class names without raising on unrecognized ones
  (Milestone 8, Issue #60).
- `examples/qiskit/`: `basic_monitor.py` (minimal integration via manual
  parameter-shift gradients, no detectors), `barren_plateau_demo.py` (the
  Qiskit version of the blueprint's Volume XX "critical MVP demo"), and
  `vqc_callback_demo.py` (wires `QiskitAdapter.callback` directly into a
  real `qiskit-machine-learning` `VQC` trainer's own `callback=` hook, with
  no manual training loop) (Milestone 8, Issue #61).
- `tests/integration/qiskit/`: end-to-end tests driving a real
  `QuantumCircuit`, a real Qiskit `Estimator` primitive, and real
  parameter-shift gradients through the full adapter -> monitor ->
  detector -> diagnosis -> action pipeline (healthy convergence, the
  engineered plateau scenario + real `StopAction` firing, circuit/optimizer
  metadata, fail-open behavior), plus a dedicated suite driving a real
  `VQC.fit()` end to end purely through `QiskitAdapter.callback()`
  (Milestone 8, Issue #62).
- `docs/integrations/qiskit.md`: Qiskit integration guide covering
  installation, the manual `record_step()`/`record_gradient()` path, all
  four `callback()` argument shapes with real `qiskit_algorithms`/
  `qiskit-machine-learning` examples, circuit/optimizer metadata
  extraction, and the version-variance handling strategy (Milestone 8,
  Issue #63). **Milestone 8 complete** ahead of the MVP release (Milestone
  7) per explicit project direction, so Qiskit integration ships alongside
  PennyLane in the same MVP.
- `qml_observer.reporting.jsonl`: JSONL event/diagnosis/summary logging.
  `JSONLWriter` appends newline-delimited JSON records (flushing on every
  write for crash-durability, per the fail-open/transparency policy,
  addendum §1); `read_jsonl()` reads them back. `event_record`/
  `diagnosis_record`/`summary_record` build the three record shapes, with
  `*_to_dict` helpers serializing every schema dataclass (enums to
  `.value`, `GradientSnapshot.values` omitted by default to keep logs
  small) (Milestone 7, Issue #48).
- `qml_observer.reporting.reporter.RunReporter`: implements the
  blueprint's Volume XII `record_event`/`record_diagnosis`/`finalize` duck
  type for `QMLMonitor(reporter=...)`, optionally streaming every record
  to a JSONL log (Issue #48) and producing a run summary dict on
  `finalize()` (idempotent). `qml_observer.reporting.summary.build_run_summary()`
  is the richer, direct-call alternative that reads circuit/optimizer/
  gradient detail from a `RunState` (e.g. `monitor.state`) -- documented
  as necessary because `QMLMonitor`'s automatic reporter hook only ever
  passes the bare `TrainingEvent`, not the full `StepObservation`
  (Milestone 7, Issue #49).
- `qml_observer.reporting.export.estimate_compute_saved()` /
  `estimate_compute_saved_from_state()`: implements the addendum §11
  resolved formula, `saved = (planned_steps - actual_steps) *
  mean_wall_time_per_step`, returning `None` (never a fabricated guess)
  when `planned_steps` or per-step timing is unavailable. Wired into both
  `RunReporter.finalize()` and `build_run_summary()`. Also adds
  `format_compute_saved()` (human-readable rendering for CLI/report
  output) and `export_summary_json()` (Milestone 7, Issue #51).
- `qml_observer.cli.main`: `qml-observer` console script (registered via
  `[project.scripts]`) with `inspect` (dump every JSONL record as pretty
  JSON) and `report` (blueprint Volume XV-style human-readable run
  summary, including status, evidence, confidence, and estimated compute
  saved) subcommands reading logs produced by `RunReporter`. `run
  config.yaml` and `benchmark <name>` are recognized but intentionally
  exit with a clear "not yet implemented" message rather than inventing an
  unspecified config/benchmark format (Milestone 7, Issue #50).

