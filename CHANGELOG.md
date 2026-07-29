# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/) once
it reaches `1.0.0`.

## [Unreleased]

Nothing yet -- see `docs/roadmap.md` for what's next (Milestone 10 onward).

## [0.2.0] - 2026-07-28

Milestone 9 complete: shot-noise-aware detection (`NoiseDetector`),
gradient-norm confidence intervals, and the calibration/reconciliation
work needed to ship both without silently shifting Milestone 7's
false-positive numbers. Also includes two Milestone 15 items pulled
forward (JSONL schema versioning, a basic performance/overhead
benchmark) per `future_milestones_plan.md`'s own recommendation to do
them now rather than at the end.

Milestone 9 (Noise & Statistical Diagnostics), Issues #64-#68:

- `qml_observer.statistics.snr`: `estimate_gradient_snr()` (Issue #64) and
  `estimate_measurement_uncertainty()` (Issue #65), the SNR/shot-noise
  primitives sketched (signature-only) in the blueprint. Handles zero
  variance/std, NaN/Inf, and non-positive `shots` explicitly, per the
  addendum §7 numerical-edge-case bar applied to every other statistics
  module. Re-exported from `qml_observer.statistics`.
- `qml_observer.detectors.noise.NoiseDetector` (Issue #66): reports
  `IssueType.NOISE_DOMINATED` when a gradient's magnitude is small only
  *relative to how few shots estimated it*, as distinct from
  `BarrenPlateauDetector`'s "genuinely collapsed" finding. Compares
  `GradientSnapshot.mean_abs` (a per-parameter magnitude) against a
  per-parameter shot-noise floor derived from `GradientSnapshot.variance`
  and the step's `shots` -- deliberately not `norm_l2`, which scales with
  `sqrt(n_parameters)` and would make the ratio insensitive to shot count
  for any circuit with more than a handful of parameters (see
  `detectors/noise.py`'s module docstring for the full rationale). Steps
  without shot-count information (`shots is None` or `<= 0`, e.g.
  analytic/adjoint execution) are skipped entirely rather than guessed at,
  so this detector never fires on a purely analytic run.
- `diagnosis/scoring.py` (Issue #67): `"noise"` is now mapped to
  `IssueType.NOISE_DOMINATED` in `ISSUE_BY_DETECTOR`, and
  `NOISE_DOMINATED` is given priority over `POSSIBLE_BARREN_PLATEAU`
  (below `CONVERGED`, which remains the highest-priority candidate) when
  both trigger for the same step -- a low-SNR reading pulls the headline
  diagnosis toward "too noisy to tell yet" rather than "confirmed
  plateau", per addendum §3's false-positive-reduction goal. Explicit
  tests pin this priority in `tests/unit/diagnosis/test_scoring.py`.
- `tests/fixtures/synthetic_runs.py` (Issue #68): new
  `finite_shots_healthy_run()`/`finite_shots_plateau_run()` generators --
  the existing `healthy_learning_run`/`artificial_plateau_run` dynamics
  with a configurable `shots` field attached to every step, so shot-budget
  behavior can be tested/benchmarked in isolation from gradient-variance
  behavior (`noise_dominated_run`, unchanged).
- `benchmarks/run_benchmarks.py` (Issue #68): `run_noise_benchmark()`
  sweeps shot budgets (`DEFAULT_SHOT_BUDGETS = (1, 5, 20, 100, 1000)`)
  across the new finite-shots fixtures with the full Milestone 9 detector
  set attached, reporting false-positive rate, plateau detection rate, and
  plateau/noise conflation rate per shot budget. Wired into
  `run_full_benchmark()`/the CLI by default (`--no-noise-benchmark` to
  skip, `--snr-threshold`/`--shot-budgets` to override).
- **Calibration finding (Issue #68):** at every shot budget from 5 upward,
  `NoiseDetector`'s placeholder `snr_threshold=1.0` produces 0% false
  positives on the finite-shots-healthy fixture and 0% conflation with
  genuine plateau detection (100% plateau detection rate, unaffected). At
  an extreme `shots=1` budget, 90% of healthy runs are (correctly)
  flagged noise-dominated, and 2% of plateau-fixture seeds have their
  headline diagnosis resolved to `NOISE_DOMINATED` alone for part of the
  run rather than `POSSIBLE_BARREN_PLATEAU`, before
  `BarrenPlateauDetector`'s own persistence window catches up -- a known,
  documented limitation at that extreme, unrealistic budget rather than a
  representative failure rate. `snr_threshold` ships unchanged based on
  this data. Full results and methodology in `docs/research/validation.md`
  and `docs/research/benchmarks.md`; raw output in
  `benchmarks/results/calibration_results.json`.
- Documentation: `docs/detectors/noise.md` filled in (was a placeholder
  stub since Milestone 7); `docs/architecture/detectors.md` and
  `docs/roadmap.md` updated to reflect Milestone 9's Issues #64-#68 as
  shipped.
- `examples/pennylane/noisy_training.py` updated to actually demonstrate
  `NoiseDetector` now that it exists, replacing its prior "this gap is
  left for Milestone 9" framing.

Milestone 9, Issue #69:

- `qml_observer.statistics.confidence`: `estimate_gradient_norm_ci()` (a
  cheap O(1) analytic delta-method interval, safe on the per-step hot
  path) and `bootstrap_gradient_norm_ci()` (a heavier, opt-in percentile-
  bootstrap alternative for offline/exploratory use, not called
  automatically anywhere). `attach_gradient_norm_ci()` composes the
  analytic interval with a `GradientSnapshot`, choosing
  `shot-noise-analytic` (via `estimate_measurement_uncertainty`) when a
  shot count is available and falling back to `parameter-spread-analytic`
  (derived from the gradient's own across-parameter `variance`) when it
  isn't -- the two are deliberately not conflated, matching the same care
  taken for `NoiseDetector`'s SNR calculation. No scipy dependency added;
  the normal quantile function uses Peter Acklam's rational approximation.
- `GradientSnapshot` (`schemas/gradient.py`) gains four new optional
  fields: `ci_lower`, `ci_upper`, `ci_level`, `ci_method`. Left unset by
  `summarize_gradient()` itself, same pattern as `snr`/`uncertainty` --
  populated later by `attach_gradient_norm_ci()`.
- `BarrenPlateauDetector` now attaches a 95% CI to its evidence on every
  step a gradient is observed (`"95% CI on gradient norm (<method>):
  [<lower>, <upper>]."`), so a "possible barren plateau" report states an
  uncertainty band rather than a bare point estimate, per Issue #69's
  goal. `docs/detectors/barren_plateau.md`'s evidence example updated to
  match.

Milestone 9, Issue #69b:

- `benchmarks/run_benchmarks.py`: `_default_detectors()` (the canonical
  detector set used by every calibration benchmark, including
  `run_calibration_sweep()`'s threshold sweep) now includes `NoiseDetector`,
  so the Milestone 7 calibration sweep actually exercises the Milestone 9
  detector set as an input, not just the finite-shots-specific
  `run_noise_benchmark()`.
- New `run_reconciliation_check()`: re-runs the Milestone 7 false-positive
  and detection-latency benchmarks with and without `NoiseDetector` in the
  detector set, against identical seeds, and reports whether any number
  changed -- addendum §3's explicit concern that "adding a new signal to
  the same deterministic scoring function can shift false-positive rates
  on the existing fixtures" checked directly rather than assumed away.
  Wired into `run_full_benchmark()`/the CLI by default
  (`--no-reconciliation-check` to skip).
- **Finding:** no change. All four Milestone 7 numbers
  (`healthy_learning`/`convergence`/`noise_dominated` false-positive
  rates; `artificial_plateau` detection rate, median/p95 steps-to-detection)
  are bit-for-bit identical with `NoiseDetector` included or excluded,
  because none of those fixtures report a `shots` field and
  `NoiseDetector` abstains on any step without one. No threshold changes
  as a result. Full results in `docs/research/benchmarks.md` and
  `docs/research/validation.md`; raw output in
  `benchmarks/results/calibration_results.json`
  (`reconciliation_check` key).
- `ScenarioRunResult` gained a `flagged_noise` field, and
  `run_false_positive_benchmark()`'s summary gained
  `n_flagged_noise_dominated` per fixture, so any future change in this
  area is visible in the benchmark output itself, not just inferred from
  the pass/fail of the false-positive target.

See `docs/roadmap.md` -- Milestone 9 (Issues #64-#69b) is now complete.

Milestone 15 (pulled forward -- see `future_milestones_plan.md`'s "Gaps &
recommendations" #5 and #7):

- **Issue #108 (JSONL schema versioning).** Every JSONL record
  (`event`/`diagnosis`/`summary`) now carries a `schema_version` field
  (`reporting/jsonl.py::JSONL_SCHEMA_VERSION`, currently `1`). Added now,
  alongside Milestone 9's new `GradientSnapshot` CI fields, rather than
  retrofitted onto a larger set of historical log shapes later. While
  implementing this, found and fixed a real bug it would have made much
  harder to diagnose after the fact: `gradient_snapshot_to_dict()` was
  missing the Issue #69 `ci_lower`/`ci_upper`/`ci_level`/`ci_method`
  fields entirely, so a "possible barren plateau" report's uncertainty
  band was silently dropped from JSONL logs even though it was present on
  the in-memory `DiagnosisResult`/`GradientSnapshot`. Fixed alongside the
  versioning work, with new tests pinning both.
- **Issue #105 (performance/overhead benchmarking), basic version.** New
  `benchmarks/run_overhead_benchmark.py`: per-step wall-clock overhead and
  peak memory for `QMLMonitor.update()` with no detectors, the full
  Milestone 9 four-detector set, and the same set with JSONL logging
  enabled, over a 2000-step run. Baseline (this environment): ~1,140
  steps/sec with the full detector set (~875µs/step), ~3,870 steps/sec
  with none (~257µs/step), ~1,060 steps/sec with logging added on top.
  No hard target set for v0.1 (same convention as Issue #54's detection-
  latency benchmark) -- a 100k+-step soak test and CI-gated regression
  tracking remain for the full Milestone 15 pass. Raw output in
  `benchmarks/results/overhead_benchmark_results.json`.

## [0.1.0] - 2026-07-25

First public MVP release (Milestone 7, Issue #57). This release includes
everything through Milestone 8: the framework-agnostic core (event
schemas, `QMLMonitor`, rolling statistics), the three MVP detectors and
diagnosis engine, the action/policy layer, JSONL logging and CLI
reporting, and *both* the PennyLane and Qiskit adapters (Milestone 8
shipped ahead of this release per explicit project direction, and all
Milestone 7 reporting/CLI/benchmark infrastructure applies identically to
it, since that layer is framework-agnostic by construction -- verified via
`examples/qiskit/barren_plateau_demo.py` producing the same
compute-saved/report output as its PennyLane counterpart).

Per the blueprint's Volume XVIII "Definition of Done" and addendum §3, the
MVP's acceptance criteria are met and documented:

- Attaches to real training loops (PennyLane and Qiskit) with minimal code
  changes.
- Detects a sustained near-zero-gradient regime (`BarrenPlateauDetector`,
  calibrated per `docs/research/validation.md`).
- Does not stop obvious convergence cases (0% false-positive rate on
  `healthy_learning`/`convergence`/`noise_dominated`, 50 seeds each).
- Produces a useful run report (`qml-observer report run.jsonl`).
- Passes the full unit + integration test suite (609 tests with both
  optional frameworks installed).
- Ships two complete end-to-end open-source examples
  (`examples/pennylane/barren_plateau_demo.py`,
  `examples/qiskit/barren_plateau_demo.py`) demonstrating the blueprint's
  Volume XX "critical MVP demo": a healthy run completing normally,
  contrasted with an engineered collapsed-gradient run stopped early with
  an estimated-compute-saved figure.

### Known limitations

See the README's "Known limitations" section for the full list. In
summary: diagnoses are probabilistic, not proof; `"pause"` currently
behaves as `"warn"` (`PauseAction` ships in Milestone 13); simulator-only
through the `0.x` series; default detector thresholds are calibrated
against the synthetic benchmark suite, not every possible circuit regime;
`QMLMonitor` is not thread-safe; and there is no automated recovery yet.

### Fixed

Found and fixed during a pre-release comprehensive review (beyond the
Milestone 7 issue list, but blocking for a beta-quality release):

- **`StagnationDetector` silently missed the common no-`parameters` case.**
  `monitor.update()`'s `parameters` argument is optional and most
  integrations (the generic-adapter/quickstart pattern) never pass it --
  but the detector required *both* loss stagnation *and* confirmed-frozen
  parameters to trigger, so a genuinely stuck run with a nonzero learning
  rate was reported as healthy indefinitely whenever the caller didn't
  also track parameters. Fixed: loss stagnation alone now triggers when
  parameters were never provided, confirmed by an additional
  least-squares-slope check (not just the raw endpoint comparison) so a
  single noisy sample can't false-trigger on an otherwise-healthy noisy
  run -- verified at 0% false-positive rate (n=100) on the
  `healthy_learning`/`convergence`/`noise_dominated` fixtures.
- **`IssueType.UNSTABLE` was never actually produced.** The issue type,
  its explanation text, and addendum §7's requirement to treat NaN/Inf
  loss as a distinct signal all existed, but no detector or the diagnosis
  engine ever checked for it -- a diverging run (loss -> NaN) was reported
  as `HEALTHY` with 100% confidence. Fixed: `DiagnosisEngine.evaluate()`
  now checks for a non-finite loss or gradient norm ahead of detector
  combination and reports `UNSTABLE`/`"critical"` immediately, overriding
  even a simultaneously-triggered `CONVERGED` reading. Added the
  `diverging_optimizer` synthetic fixture and dedicated unit/fixture-level
  tests for this path (previously zero coverage of any NaN/Inf scenario
  existed anywhere in the suite).
- Suppressed the numpy `RuntimeWarning` from computing gradient
  norm/variance over an Inf/NaN-containing array (`summarize_gradient`) --
  an anticipated case (addendum §7), not a real numerical bug, matching
  the existing suppression in `statistics.loss.relative_loss_improvement`.
- Closed a test-coverage gap: no test previously exercised
  `build_run_summary()`'s gradient/circuit fields actually rendering
  through `qml-observer report` (`RunReporter`'s own automatic summary
  never includes them -- see `reporting/reporter.py`'s docstring -- so
  this path was silently unverified end-to-end).

### Added
- `qml_observer.telemetry`: opt-in, anonymized telemetry (addendum §5),
  disabled by default. `telemetry.enable()`/`disable()`/`is_enabled()`
  and `qml-observer telemetry {enable,disable,status}` manage consent,
  persisted to `~/.config/qml-observer/telemetry.json`; a non-interactive
  environment (no TTY) is never auto-enrolled. `TelemetryCollector`
  builds an anonymized `TelemetryRecord` (detector names, extracted
  numeric thresholds, diagnosis issue/confidence, framework label, a
  coarse qubit-count bucket, detection-latency steps -- never raw
  gradients, loss, circuit structure, parameters, run IDs, file paths, or
  hostnames) and either queues it locally as JSON Lines or POSTs it to an
  explicitly configured endpoint; this release ships no bundled backend.
  `QMLMonitor(telemetry_collector=..., telemetry_framework=...)` wires it
  in end-to-end, fully opt-in and fail-open (a broken/misconfigured
  collector can never affect `finish()` or the training loop). Full
  schema published in `docs/development/telemetry.md`
  (Issue #9b/#9c).
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
- `benchmarks/run_benchmarks.py`: the runnable calibration/benchmark
  harness -- `run_false_positive_benchmark()` (Issue #53, healthy/
  convergence/noise-dominated fixtures), `run_detection_latency_benchmark()`
  (Issue #54, artificial-plateau fixture), `run_full_benchmark()` (Issue
  #55, the combined comparison), and `run_calibration_sweep()` (addendum
  §3's threshold-sweep primitive). Reuses
  `tests/fixtures/synthetic_runs.py` rather than a second fixture set.
  Results are saved to `benchmarks/results/calibration_results.json`
  (Milestone 7, Issues #53-#55).
- `benchmarks/qml_observer_benchmark.ipynb`: the benchmark notebook
  (Milestone 7, Issue #52) -- a narrative, executed wrapper around
  `run_benchmarks.py` covering the calibration sweep, false-positive/
  detection-latency results, and the live PennyLane "critical MVP demo".
- **Calibration finding (Issue #54/#55b):** the benchmark suite found
  `BarrenPlateauDetector`'s original `gradient_threshold=1e-8` placeholder
  never triggered on the `artificial_plateau` fixture at all (0%
  detection rate across 50 seeds) -- that fixture's collapsed-gradient
  scale (`~1e-6`) never fell below `1e-8`. Recalibrated to `5e-6`: 0%
  false-positive rate on `healthy_learning`/`convergence`/`noise_dominated`
  (50 seeds each) and 100% detection on `artificial_plateau` (median 14 /
  p95 21 steps-to-detection). `variance_threshold`'s default moved
  accordingly (`gradient_threshold ** 2`). See
  `docs/research/validation.md` and `docs/research/benchmarks.md` for the
  full methodology, sweep table, and versioned record of this change
  (Milestone 7, Issue #54/#55b).
- MVP documentation (Milestone 7, Issue #56): `docs/index.md`;
  `docs/getting_started/{installation,quickstart,concepts}.md` (including
  the "How to interpret alerts" guide); `docs/architecture/{overview,
  events,adapters,detectors,actions}.md`; `docs/detectors/{barren_plateau,
  stagnation,convergence,noise}.md`; `docs/integrations/{pennylane,
  generic}.md` (`qiskit.md` shipped with Milestone 8); `docs/research/
  {methodology,benchmarks,validation}.md`; `docs/development/
  {contributing,development_setup,adding_detectors}.md`; `docs/roadmap.md`
  (documenting addendum §4's simulator-only scope and what hardware
  integration would need).

