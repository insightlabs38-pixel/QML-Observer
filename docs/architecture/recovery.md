# Recovery engine (Milestone 13)

Covers the first five issues of the updated Milestone 13 from
`future_milestones_plan.md`, in the sequence that document requires:
**#90b, #90, #91, #92, #93**, plus the remaining Milestone 13 issues
(**#94, #95, #96, #96b, #97**) completed in the same pass. #90b is a
prerequisite for #97 (automatic resume) and therefore ships first, ahead
of the blueprint's original numeric order.

Milestone 13 status: **complete -- Issues #90b, #90-#97 (including #96b)
all shipped.**

---

## Issue #90b -- `PauseAction` (`actions/pause.py`)

### What it does

Gives `"pause"` real, distinct behavior for the first time: previously it
behaved identically to `"warn"` (a deliberate conservative placeholder,
per `docs/architecture/actions.md`'s pre-Milestone-13 text). `PauseAction`
records a pause request via a `.triggered`/`.paused` flag -- mirroring
`StopAction`'s "the caller's own loop must check and act" contract, never
reaching into the loop directly (plan.md §2, non-invasive core principle)
-- and captures a `PausedRunSnapshot` (run ID, step, window size,
`planned_steps`, and the triggering diagnosis) so a future `RecoveryExecutor
`-driven resume (Issue #97) has something concrete to reconstruct a
`QMLMonitor` from.

`QMLMonitor.update()`/`finish()` now pass their own run context
(`run_id`, `state.step_count`, `window_size`, `planned_steps`) into
`ActionPolicy.execute()`, which forwards it to `PauseAction.execute()`
when selected, so the captured snapshot is genuinely populated rather
than an empty placeholder. `QMLMonitor.should_pause()` mirrors
`should_stop()`'s pure, side-effect-free recomputation from
`state.latest_diagnosis`.

Degraded-diagnosis safety (addendum §1) applies identically to
`StopAction`: a `degraded=True` diagnosis never selects `PauseAction`
unless `mode="adaptive"` **and** `allow_stop_on_degraded=True` -- though
in practice `"adaptive"`/`"stop"` never select `PauseAction` at all (they
escalate straight past pausing to stopping); only `mode="pause"` can
select it.

### Validation

`tests/unit/policies/test_pause.py` (unit-level: triggering, snapshot
capture with/without run context, resume-vs-reset semantics, fail-open
logging) and `tests/unit/core/test_action_integration.py::TestPauseMode`
(integration-level, through a real `QMLMonitor` + `BarrenPlateauDetector`
run). `tests/unit/policies/test_policies.py::TestPauseMode` covers
`ActionPolicy.select_action()`'s mode/severity/degraded matrix directly.

### Known limitations

- `PauseAction` cannot pause the caller's actual training loop -- it can
  only signal via `.triggered`/`should_pause()`. The caller is
  responsible for checking it between steps, same as `should_stop()`.
- `PausedRunSnapshot` does not (yet) capture parameter values, optimizer
  internal state, or RNG state -- only the metadata needed to
  reconstruct an equivalent `QMLMonitor`'s *configuration*, not the
  underlying training state. Capturing/restoring the latter is explicitly
  the caller's own responsibility until Issue #97 defines what "automatic
  resume" needs beyond this.

---

## Issue #90 -- Recovery strategy interface (`qml_observer.recovery`)

### What it does

New `qml_observer.recovery` package, deliberately separate from
`qml_observer.actions` (recovery is opt-in and never auto-wired into
`ActionPolicy`, per the blueprint's explicit "do not implement automatic
recovery until the detection system is validated," Volume XIV):

- `RecoveryContext` -- circuit/optimizer/shots/gradient/`planned_steps`
  context a strategy may need, mirroring the fields already threaded
  through `core.events.StepObservation`.
- `RecoveryRecommendation` -- one strategy's ranked, concrete proposal
  (`priority` in `[0, 1]`, `parameters`, `rationale`, an optional
  `hook_name` naming the method a `RecoveryExecutor` will look for).
- `RecoveryOutcome` -- mirrors `actions.base.ActionResult`'s
  `executed`/`message` shape for the recovery layer's own
  `applied`/`message`.
- `RecoveryStrategy` (ABC) -- `applies_to(diagnosis)` / `propose(diagnosis,
  context)`, the same "cheap filter, then do the work" split
  `BaseDetector` uses.
- `RecoveryPlanner.recommend(diagnosis, context, allow_degraded=False)` --
  asks every applicable strategy to `propose()`, sorts by `priority`
  descending, catches and logs (not raises) any exception an individual
  strategy produces, and refuses to produce recommendations for a
  `degraded=True` diagnosis unless explicitly overridden -- the recovery
  layer's own version of addendum §1's conservative default.
- `RecoveryExecutor.apply(recommendation, training_state)` -- looks for
  `recommendation.hook_name` on the caller-supplied `training_state`
  object; calls it with `recommendation.parameters` if present and
  callable, reports "not applied, do this manually" otherwise, and never
  raises (a hook that itself raises is caught and reported, same
  fail-open contract as `Action.execute()`).

### Validation

`tests/unit/recovery/test_recovery_base.py` (dataclass validation,
`RecoveryStrategy` ABC contract), `test_planner.py` (ranking order,
degraded suppression/override, per-strategy fail-open behavior via
strategies engineered to raise), `test_executor.py` (hook dispatch,
missing/non-callable hook handling, hook-raises fail-open behavior).

### Known limitations

- `RecoveryPlanner`/`RecoveryExecutor` have no built-in evaluation loop
  ("test a small number of changes, resume only if health metrics
  improve" per plan.md §22) -- that is Issue #96, not yet implemented.
  Calling `RecoveryExecutor.apply()` today is a one-shot action with no
  automatic before/after health comparison.
- `RecoveryContext` has no `RecoveryContext.from_state(...)` convenience
  constructor yet; callers build it from their own `RunState`/
  `StepObservation` data by hand. Worth adding once a concrete caller
  (e.g. the CLI or a future orchestration layer) needs it.

---

## Issue #91 -- `ParameterReinitializationStrategy`

### What it does

Applicable to `POSSIBLE_BARREN_PLATEAU` and `STAGNATION` (both are
"training looks stuck at its current trajectory" verdicts). For a barren
plateau with a generic `CircuitMetadata.initialization` (e.g.
`"random_uniform"`, or unknown), it specifically recommends a
`"reduced_domain"`/small-angle reinitialization; for a barren plateau
where the initialization already looks barren-plateau-aware, it still
proposes a plain reinit but at markedly lower priority, since the likelier
culprit is elsewhere (circuit depth/entanglement -- not this strategy's
scope). For stagnation, it proposes a plain reinit to escape a poor local
trajectory, independent of initialization style.

### Mathematical / scientific description

Small-angle (near-identity) initialization keeps a parameterized circuit
close to the identity operation at the start of training, away from the
concentration-of-measure regime in which cost-function gradients
concentrate exponentially around zero for generic, sufficiently expressive
deep circuits (the barren-plateau phenomenon `docs/detectors/
barren_plateau.md` already documents at the detector level). This
strategy does not re-derive that result -- it applies it as a documented,
literature-grounded mitigation once the *detector* has already flagged the
symptom.

### References

- McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H.
  (2018), *Barren plateaus in quantum neural network training landscapes*
  -- the original characterization of the exponential gradient-vanishing
  phenomenon this strategy's rationale is grounded in.
- Grant, E., Wossnig, L., Ostaszewski, M., & Benedetti, M. (2019), *An
  initialization strategy for addressed barren plateaus in parametrized
  quantum circuits* -- the specific reduced-domain/identity-block
  initialization mitigation this strategy recommends.

### Validation

`tests/unit/recovery/test_reinitialization.py`: `applies_to()` matrix
across all `IssueType` values, generic-vs-aware-initialization priority
ordering, parameter shape for each branch, priority always in `[0, 1]`
across the full confidence range.

### Known limitations

- This strategy only ever *recommends* a reinitialization strategy name
  (`"reduced_domain"`) via `RecoveryRecommendation.parameters`; it does
  not implement the reduced-domain sampling itself. A `training_state.
  reinitialize_parameters(initialization="reduced_domain")` hook must
  interpret that string -- qml_observer does not own parameter sampling.
- The generic-vs-aware classification of `CircuitMetadata.initialization`
  is a small fixed set of known strings (`_GENERIC_INITIALIZATIONS`);
  an adapter reporting an unrecognized custom initialization name is
  currently treated as non-generic (lower priority), which may
  under-recommend for adapters using nonstandard naming. Worth revisiting
  once real adapter data on `initialization` naming conventions exists.

---

## Issue #92 -- `LearningRateAdjustmentStrategy`

### What it does

Applicable to `UNSTABLE` (recommends halving the learning rate) and
`STAGNATION` (recommends doubling it). Deliberately **not** applicable to
`POSSIBLE_BARREN_PLATEAU`: scaling an already-vanishing gradient by a
larger step still vanishes, so a learning-rate change is not a
mechanistically appropriate remedy for that issue type --
`ParameterReinitializationStrategy` (Issue #91) is the better-targeted
strategy there.

### Mathematical / scientific description

For gradient descent `theta_{t+1} = theta_t - lr * grad`, oscillation or
divergence (`UNSTABLE`) is the textbook symptom of a step size exceeding
the local Lipschitz-smoothness bound of the loss landscape; halving `lr`
is the standard first response. Conversely, an "effectively frozen"
optimizer (`STAGNATION`, blueprint Volume VI-2) with a healthy (non-tiny)
gradient signal can mean the current step size is too small to make
visible progress relative to the loss landscape's local curvature;
doubling `lr` is a correspondingly modest first response. Both multipliers
(`0.5x` / `2.0x`) are placeholder heuristics, not calibrated constants --
see Known Limitations.

### Validation

`tests/unit/recovery/test_learning_rate.py`: `applies_to()` matrix,
known-vs-unknown-vs-zero current learning rate for both directions,
priority ordering (instability ranked above stagnation at equal
confidence, reflecting instability being the stronger, more directly
actionable signal).

### Known limitations

- The `0.5x`/`2.0x` multipliers are fixed constants, not derived from the
  observed loss curvature or gradient statistics -- unlike
  `ShotBudgetAdjustmentStrategy`'s SNR-derived multiplier (Issue #93),
  this strategy does not yet use `RecoveryContext.gradient` at all. A
  future revision could scale the adjustment by observed instability
  severity (e.g. loss divergence rate) rather than a fixed factor.
  Per addendum §3's empirical-calibration precedent, these should be
  treated as placeholders pending benchmark-driven tuning, not final
  values.
- No lower/upper bound is enforced on the proposed `learning_rate`
  besides staying positive; a caller applying many successive stagnation
  recommendations without ever pausing to re-diagnose could in principle
  compound the `2.0x` multiplier indefinitely. `RecoveryExecutor.apply()`
  is a one-shot operation with no history of prior recommendations, so
  guarding against this is currently the caller's responsibility (see
  Issue #96, recovery evaluation, not yet implemented).

---

## Issue #93 -- `ShotBudgetAdjustmentStrategy`

### What it does

Applicable only to `NOISE_DOMINATED` -- the diagnosis Milestone 9's
`NoiseDetector` reports when a gradient estimate is statistically
indistinguishable from shot noise rather than genuinely small. Computes a
concrete target shot count from the run's current shot count and gradient
SNR, when both are available in `RecoveryContext`; falls back to a fixed
generic starting budget (`4096` shots) when they are not.

### Mathematical description

Shot-noise uncertainty on an expectation-value estimate scales as
`uncertainty ~ 1 / sqrt(shots)` (`statistics.snr.
estimate_measurement_uncertainty`, already used by `NoiseDetector`). Since
SNR is inversely proportional to that uncertainty:

```
snr(shots) ~ sqrt(shots)
snr_target / snr_now = sqrt(shots_target / shots_now)
shots_target = shots_now * (snr_target / snr_now) ** 2
```

`snr_target` defaults to `1.5` (`_DEFAULT_TARGET_SNR`) -- just above
`NoiseDetector`'s own default `snr_threshold` of `1.0`
(`detectors/noise.py`), so the recommended shot count clears the
detection threshold with a small margin rather than landing exactly on
it. The computed multiplier is capped at `100x`
(`_MAX_SHOT_MULTIPLIER`) so a pathologically low (near-zero) observed SNR
cannot propose an unbounded, budget-destroying shot count, and the
strategy never proposes *fewer* shots than currently in use.

### References

- The `1/sqrt(shots)` shot-noise scaling is standard quantum measurement
  statistics (binomial/multinomial sampling variance of Pauli
  expectation-value estimators); no single citation is needed beyond
  what `statistics/snr.py` and `docs/detectors/noise.md` already
  document for `NoiseDetector` itself, which this strategy's math directly
  builds on.

### Validation

`tests/unit/recovery/test_shot_budget.py`: low-SNR context recommends
more shots, already-high-SNR context never recommends fewer, near-zero/
zero-SNR context hits the multiplier cap without a divide-by-zero error,
missing-shots/missing-gradient contexts fall back to the generic
recommendation rather than raising.

### Known limitations

- The `shots_target` formula assumes per-shot variance stays
  approximately constant as the shot count changes for a *fixed* circuit
  and observable -- true for re-measuring the same circuit at the same
  parameters, but not a guarantee if the optimizer has also moved to a
  meaningfully different point in parameter space by the time a
  recommendation is actually applied.
- The `100x` cap is a safety bound, not a calibrated value -- like Issue
  #92's fixed multipliers, this is a placeholder pending the addendum
  §3-style empirical calibration process, not a final constant.

---

## Issue #94 -- `AnsatzDepthReductionStrategy`

### What it does

Applicable only to `POSSIBLE_BARREN_PLATEAU`. Unlike
`ParameterReinitializationStrategy` (targets the starting point), this
strategy targets circuit *expressivity* directly, halving `CircuitMetadata
.depth` by default. Returns `None` (rather than inventing a depth) when
`context.circuit`/`.depth` is unavailable, or when depth is already at the
`_MIN_DEPTH` floor -- `RecoveryPlanner` simply excludes the strategy from
that recommendation set rather than receiving a meaningless proposal.

### Mathematical / scientific description

Barren-plateau severity is theoretically predicted, and empirically
confirmed by `qml_observer.advanced.scaling.ScalingAnalyzer.
analyze_depth_scaling` (Milestone 12, Issue #89) on a per-run basis, to
worsen with circuit depth for sufficiently expressive, hardware-efficient
ansaetze -- gradient variance decays consistent with an exponential
function of depth. Reducing depth is therefore a mechanistically direct
mitigation, not a workaround, unlike a learning-rate or optimizer change.

### Validation

`tests/unit/recovery/test_depth_reduction.py`: `applies_to()` matrix,
default/custom reduction fractions, floor behavior, missing-circuit/
missing-depth returns `None`, priority boost for circuits past the
"clearly deep" threshold.

### Known limitations

- `reduction_fraction` (default `0.5`) is a fixed heuristic, not derived
  from the observed depth-scaling fit `ScalingAnalyzer` could in
  principle supply -- a future revision could size the reduction from
  `ScalingAnalysisResult`'s fitted decay rate rather than a constant
  50%, but that would require threading Milestone 12's scaling analysis
  into `RecoveryContext`, not yet done.
- This strategy only proposes a target depth; it cannot itself decide
  *which* gates/layers to remove -- that interpretation is entirely
  `training_state.set_circuit_depth()`'s responsibility, whatever that
  means for the caller's specific ansatz construction.

---

## Issue #95 -- `OptimizerSwitchingStrategy`

### What it does

Applicable to `UNSTABLE` and `STAGNATION`. Classifies the current
optimizer (from `OptimizerMetadata.name`) into one of three families --
adaptive (Adam-family), conservative (plain gradient descent), or
perturbation-based (SPSA-family) -- and recommends switching direction
based on the diagnosis: away from adaptive toward conservative (or
conservative toward perturbation-based) for instability; toward adaptive
(or adaptive toward perturbation-based) for stagnation. Deliberately
excludes `POSSIBLE_BARREN_PLATEAU`: switching which classical optimizer
processes the gradient signal does not address a vanishing gradient
signal itself.

### Validation

`tests/unit/recovery/test_optimizer_switching.py`: full switching-matrix
coverage for both diagnosis directions across all three known families
plus an unknown-optimizer-name fallback, case-insensitive name matching,
missing-optimizer-context fallback, priority ordering (instability ranked
above stagnation at equal confidence).

### Known limitations

- The three-family classification (`_ADAPTIVE_OPTIMIZERS`/
  `_CONSERVATIVE_OPTIMIZERS`/`_PERTURBATION_OPTIMIZERS`) is a small fixed
  set of known optimizer-name strings; an adapter reporting an
  unrecognized name is classified `"unknown"` and handled via a
  conservative fallback branch, not misclassified into the wrong family,
  but the specific optimizer suggested in that branch (`SPSA`) is a
  generic default rather than tailored to whatever the unrecognized
  optimizer actually is.
- No interaction with `LearningRateAdjustmentStrategy` is modeled: a
  `RecoveryPlanner` configured with both strategies will propose both an
  optimizer switch and a learning-rate change independently for the same
  diagnosis, with no guidance on whether to apply both simultaneously or
  one at a time. `RecoveryEvaluator` (Issue #96) evaluates whichever
  single change was actually applied; comparing *combinations* of
  recommendations is out of scope for both strategies and the evaluator.

---

## Issue #96b -- `NaturalGradientStrategy`

### What it does

Applicable to `POSSIBLE_BARREN_PLATEAU` and `STAGNATION`. Recommends
switching to a quantum-natural-gradient-aware optimizer
(`QuantumNaturalGradient`), which preconditions the ordinary gradient by
the inverse quantum Fisher information matrix (QFIM,
`qml_observer.advanced.geometry.qfim`, Milestone 12) rather than treating
parameter space as flat. Returns `None` if the current optimizer is
already recognized as natural-gradient-aware (e.g. `"QNSPSA"`).
Deliberately capped at a lower priority ceiling (`<= 0.4`) than the
cheaper strategies above it, since QFIM estimation is a per-step cost
that grows with circuit size -- this is a "reach for after cheaper
interventions" recommendation, not a first response.

### References

- Stokes, J., Izaac, J., Killoran, N., & Carleo, G. (2020), *Quantum
  Natural Gradient* -- the QFIM-as-preconditioner formulation this
  strategy's recommendation is grounded in; the same reference already
  cited for Milestone 12's QFIM estimation (`docs/research/geometry.md`,
  Issue #83).

### Validation

`tests/unit/recovery/test_natural_gradient.py`: `applies_to()` matrix,
already-natural-gradient-optimizer short-circuit (case-insensitive),
priority cap enforcement, QFIM-cost rationale note when circuit size is
known.

### Known limitations

- This strategy does not consult `qml_observer.advanced.geometry.qfim_
  condition_number`/`effective_rank` to decide *whether* natural gradient
  would actually help this specific circuit's QFIM conditioning -- it
  recommends the method generically whenever the issue type applies and
  the optimizer isn't already natural-gradient-aware. A future revision
  could raise or lower priority based on an actual QFIM conditioning
  estimate, at the cost of requiring one to have already been computed
  (an expensive, opt-in Milestone 12 operation) before recovery planning
  can run.

---

## Issue #96 -- Recovery evaluation (`RecoveryEvaluator`)

### What it does

Implements plan.md §22's "test a small number of changes, resume only if
health metrics improve." `RecoveryEvaluator.evaluate(strategy_name,
before, after)` compares two `DiagnosisResult`s (typically: the diagnosis
that triggered a pause, and a later diagnosis observed after resuming and
applying a recovery recommendation) and returns a
`RecoveryEvaluationResult` with `improved`/`conclusive`/`summary`.
`should_keep(result)` is the keep-vs-roll-back decision plan.md §22 asks
for.

Comparison rules, in priority order: (1) a degraded `before` or `after`
is always inconclusive (addendum §1 -- unreliable evidence cannot judge
success); (2) reaching `HEALTHY`/`CONVERGED` from anything else is
unambiguous improvement; (3) regressing *out of* `HEALTHY`/`CONVERGED` is
unambiguous non-improvement; (4) same issue, before/after: compare
`SEVERITY_RANK` first, then (if unchanged) require a confidence drop of
at least `confidence_improvement_threshold` (default `0.1`) to call it
improved, rather than "unchanged, within noise"; (5) different issue,
neither a good-state transition: compare `SEVERITY_RANK` as the
tie-breaker.

### Validation

`tests/unit/recovery/test_evaluation.py`: full coverage of all five
comparison branches above, both directions of the good-state transition,
confidence-threshold boundary behavior, `should_keep()`'s conservative
default (never keeps an inconclusive result).

### Known limitations

- `RecoveryEvaluator` only compares two point-in-time diagnoses; it has
  no notion of *how many* steps elapsed between `before` and `after`, or
  whether that was "enough" steps for the comparison to be meaningful
  (e.g. a detector's own `patience` window). The caller decides when to
  call `evaluate()`, same as the caller decides when to call
  `RecoveryExecutor.apply()` in the first place.
- The same-issue confidence-threshold comparison (branch 4 above) treats
  every issue type's confidence scale as directly comparable at a fixed
  threshold; different detectors could in principle produce confidence
  scores with different practical "noise floors". No per-issue-type
  threshold calibration exists yet (would follow the same addendum
  §3-style empirical process as everything else calibration-related in
  this project).

---

## Issue #97 -- Automatic resume (`resume_monitor_from_snapshot`)

### What it does

`qml_observer.recovery.resume.resume_monitor_from_snapshot(snapshot, ...)`
reconstructs a `QMLMonitor` from a `PausedRunSnapshot` (Issue #90b): same
`run_id`, `window_size`, and `planned_steps`, with its step counter seeded
via the new `RunState.seed_step_count()` so the next `update()` call
continues the step sequence at the correct number. Stated plainly, per
the module's own docstring: this is "automatic resume" of *monitoring*,
not of the caller's actual quantum computation -- qml_observer does not
own the training loop (plan.md §2) and never has; reconstructing a
monitor's configuration is exactly as far as a non-invasive observability
layer can honestly automate this.

`RunState.seed_step_count()` is deliberately narrow: it only accepts a
non-negative step count and only on a `RunState` with zero recorded
observations (fresh or freshly-`reset()`), raising otherwise -- it exists
specifically to support this resume path, not as a general-purpose
rewind/fast-forward API, and calling it after real observations exist
would silently misrepresent the window's actual history.

### Validation

`tests/unit/recovery/test_resume.py`: reconstruction preserves `run_id`/
`window_size`/`planned_steps`, seeds the step counter, leaves the window
empty, threads through custom `policy`/`action_policy`/`detectors`,
handles the `window_size=0`/`run_id="unknown"` edge cases a
context-free `PauseAction.execute(diagnosis)` call would produce, and an
end-to-end test: a real `QMLMonitor` + `BarrenPlateauDetector` run pauses,
its snapshot is used to reconstruct a fresh monitor, and that monitor
correctly continues the step sequence and produces diagnoses for new data.
`tests/unit/core/test_state.py::TestSeedStepCount` covers
`seed_step_count()` directly: fresh-state success, negative-step
rejection, already-has-observations rejection, and allowed-again-after-
`reset()`.

### Known limitations

- The rolling window of prior `StepObservation`s (recent gradients/loss
  values) is **not** restored on resume -- `PausedRunSnapshot`
  deliberately does not capture it (not a lightweight, serializable-
  friendly field). Any detector relying on `patience`/persistence across
  the pause boundary needs that many new steps again post-resume before
  it can trigger; this is the direct consequence of `PausedRunSnapshot`'s
  own scope decision (Issue #90b), not a new limitation introduced here.
- Wall-clock duration is not preserved across the pause: the resumed
  monitor's `start_time` is set on its next `update()` call (auto-start),
  not backdated to the original run's start, so a compute-saved estimate
  computed after resuming reflects only time elapsed after the resume.
- `resume_monitor_from_snapshot()` does not itself call
  `RecoveryPlanner`/`RecoveryExecutor`/`RecoveryEvaluator` -- it only
  handles the monitor-reconstruction step. A full pause -> recover ->
  resume -> evaluate loop is composed by the caller from these pieces;
  no single orchestrating function exists (or is planned) that drives a
  caller's actual training loop end-to-end, consistent with the
  non-invasive core principle this entire module is scoped around.
