# Research-grade geometry & scaling diagnostics (Milestone 12)

Covers Issues #83-#89 (`qml_observer.advanced.geometry` and
`qml_observer.advanced.scaling`) -- all seven issues originally scoped
for Milestone 12 in `future_milestones_plan.md`. Per Issue #89b, every
issue in this milestone must ship with the blueprint's Volume XVIII
"Definition of Done" bar for research features -- mathematical
description, references, validation methodology, benchmark results, and
known limitations -- documented here as a hard gate, not a follow-up.

Milestone 12 status: **Issues #83-#89 complete.** Issue #89b's own
"documentation requirement as a hard gate per issue" is satisfied by this
page for all seven; its broader ask ("treat as a hard gate," i.e. don't
defer any of it) is addressed by writing this alongside the code rather
than after.

---

## Issue #83 -- QFIM estimation (`qfim.py`)

### Mathematical description

For a pure state `|psi(theta)>`, the quantum Fisher information matrix
(QFIM) is the Fubini-Study metric tensor:

```
F_ij(theta) = 4 * Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]
```

`estimate_qfim(state_fn, parameters, eps)` estimates `d_i psi` by central
finite differences of a caller-supplied, framework-agnostic
`state_fn: parameters -> statevector`, then assembles `F` from those
estimates, symmetrizing the result to cancel floating-point asymmetry.

### References

- Meyer, J. J. (2021), *Fisher Information in Noisy Intermediate-Scale
  Quantum Applications*, for the QFIM/Fubini-Study background this
  implementation follows.
- Stokes et al. (2020), *Quantum Natural Gradient*, for the broader
  motivation (QFIM as the natural-gradient preconditioner) and for the
  analytic parameter-shift QFIM rule this module does **not** implement
  (see Known Limitations).

### Validation methodology

`tests/unit/advanced/geometry/test_qfim.py` checks the estimator against
three closed-form reference cases rather than only checking internal
consistency:

1. A single-qubit `RY(theta)` rotation, whose QFIM is analytically `1`
   everywhere (independent of `theta`) in these units.
2. Two independent single-qubit `RY` rotations (a product state), whose
   QFIM is analytically diagonal with both entries `1` (no cross terms).
3. A two-parameter state depending only on `theta0 + theta1`, whose QFIM
   is analytically rank-1 (singular along `theta0 - theta1`) -- checking
   that the estimator correctly detects a *known* redundant direction,
   not just that it produces *some* PSD matrix.

All three pass at `abs=1e-3` tolerance with the default `eps=1e-4`
finite-difference step. General PSD-ness (all eigenvalues `>= -1e-8`) is
also checked on a non-degenerate case, since finite-difference asymmetry
could otherwise mask a negative eigenvalue as a real effect rather than
numerical error.

### Benchmark results

Not yet run against a real PennyLane/Qiskit statevector simulation (only
the closed-form reference cases above); see Known Limitations.

### Known limitations

- **Finite-difference only.** No analytic parameter-shift QFIM rule
  (Stokes et al. 2020) is implemented; for ansätze built entirely from
  single-parameter Pauli rotations, an analytic block-diagonal QFIM would
  be both cheaper and exact rather than approximate. This is documented
  future work, not implemented here, to keep the first version
  independent of any specific gate-set assumption.
- **Requires an exact/analytic `state_fn`.** Finite differences of a
  finite-shots-sampled state estimate would amplify shot noise into a
  very unreliable QFIM; `estimate_qfim` documents this requirement but
  cannot enforce it (it has no way to know whether the caller's
  `state_fn` is analytic or sampled).
- **Not benchmarked against a real framework's statevector output** (only
  hand-constructed closed-form reference states in the unit tests above).
  A PennyLane/Qiskit-based validation notebook is future work, matching
  how `examples/{pennylane,qiskit}/barren_plateau_demo.py` complements
  the Milestone 7 synthetic-fixture calibration.

---

## Issue #84 -- QFIM conditioning (`conditioning.py`)

### Mathematical description

Given the QFIM's eigenvalues `lambda_1 >= ... >= lambda_n >= 0`:

- `qfim_condition_number = lambda_1 / lambda_n` (`inf` if `lambda_n` is
  numerically zero relative to `lambda_1`, per an `rcond` threshold,
  rather than a large-but-arbitrary finite value).
- `effective_rank = count(lambda_i >= threshold * lambda_1)`, a simple
  relative-threshold soft rank (default `threshold=1e-6`).

### References

Roy, O. & Vetterli, M. (2007), *The effective rank: A measure of
effective dimensionality*, for the general concept of a soft/effective
rank as an alternative to the threshold definition used here (see Known
Limitations for why the simpler definition was chosen for this first
version).

### Validation methodology

`tests/unit/advanced/geometry/test_conditioning.py` uses hand-constructed
diagonal QFIMs with known eigenvalues (condition number and rank are
then simple arithmetic to verify by hand), plus edge cases: an all-zero
QFIM (`condition_number = inf`, `effective_rank = 0`), a near-singular
QFIM below `rcond`, and a boundary case exercising the relative-threshold
definition of `effective_rank` directly (`threshold=0.005` vs.
`threshold=0.5` on the same `diag(100, 1)` QFIM, verifying the threshold
is applied relative to `lambda_max`, not as an absolute cutoff).

### Benchmark results

Not applicable -- this is a deterministic linear-algebra computation with
no tunable default requiring empirical calibration against training
fixtures (unlike, e.g., `BarrenPlateauDetector`'s thresholds).

### Known limitations

- **Threshold-based, not entropy-based, effective rank.** The simpler
  relative-threshold definition was chosen over Roy & Vetterli's entropy-
  based effective rank for auditability (a single, explainable
  comparison per eigenvalue) at the cost of being a genuinely different
  number in general; both are legitimate "soft rank" definitions and the
  choice should be revisited if downstream use (e.g. a future redundancy-
  aware detector) shows the threshold definition is too sensitive to its
  cutoff in practice.
- **`rcond`/`threshold` defaults are placeholders**, not empirically
  calibrated the way Milestone 7's detector thresholds were (addendum
  §3) -- there is no equivalent benchmark suite here yet because there is
  no real-circuit QFIM benchmark corpus (see Issue #83's benchmark gap
  above).

---

## Issue #85 -- Parameter-redundancy detection (`redundancy.py`)

### Mathematical description

For each QFIM eigenvector `v_k` associated with a near-zero eigenvalue
(per `effective_rank`'s threshold), flag parameter index `i` as a
redundancy candidate if `|v_k[i]|^2 >= contribution_threshold` (default
`0.1`) -- eigenvector components are unit-normalized, so squared
components are directly interpretable as "fraction of this null-space
direction's norm attributable to parameter `i`."

### References

No dedicated external reference beyond the QFIM/effective-rank references
above (#83, #84); this is a straightforward application of standard
null-space analysis (which components of a near-zero eigenvector are
large) to the redundancy-detection question, not a technique reproduced
from a specific paper.

### Validation methodology

`tests/unit/advanced/geometry/test_redundancy.py` checks four
hand-constructed cases with a known ground-truth answer: a full-rank QFIM
(no redundancy), a QFIM with one parameter's row/column exactly zero
(that exact parameter flagged, nothing else), a QFIM with a *coupled*
null direction `(1,-1)/sqrt(2)` on two parameters (both flagged, matching
the `theta0 + theta1`-only dependency case from Issue #83's QFIM test),
and a fully degenerate (all-zero) QFIM (every parameter flagged, since
"no redundancy found" would misreport a fully-degenerate result as
healthy).

### Benchmark results

Not yet run against a real ansatz with a known, literature-documented
redundant-parameter pair (e.g. two consecutive commuting rotations); see
Known Limitations.

### Known limitations

- **Local, not global.** A parameter flagged here is redundant *at the
  sampled parameter point*, not necessarily everywhere in parameter
  space -- a different point could have a full-rank QFIM for the same
  ansatz. This is stated explicitly in the module docstring and should
  be repeated in any user-facing report built on top of this function.
- **`contribution_threshold=0.1` is a placeholder default**, not
  empirically tuned; the closest analogue to Milestone 7's calibration
  process (varying the threshold against seeded fixtures with known
  ground truth) has not yet been run for this module.
- **Not validated on a real multi-qubit ansatz** with more than two
  parameters or a non-trivial gate structure (only the small closed-form
  cases above); this is the most important gap to close before relying
  on this function's output to guide real circuit redesign decisions.

---

## Issue #86 -- Hessian-vector product estimation (`hessian.py`)

### Mathematical description

`estimate_hessian_vector_product(loss_fn, parameters, vector)` estimates
`H(parameters) @ vector` via a nested finite difference: a central
difference of two finite-difference gradient estimates,

```
Hv ~= (grad(theta + eps*v) - grad(theta - eps*v)) / (2*eps)
```

where `grad` itself is estimated by per-parameter central finite
differences of `loss_fn` (no analytic gradient function required, per the
blueprint's exact signature and its "observe, don't reimplement the
framework's differentiation machinery" philosophy).

### References

Pearlmutter, B. A. (1994), *Fast Exact Multiplication by the Hessian* --
the classic *analytic* HVP trick, requiring an autodiff/analytic gradient
function, deliberately **not** used here (see module docstring for why:
it would mean depending on or reimplementing the specific framework's
differentiation machinery). Cited as the reason a pure finite-difference
fallback was chosen instead, not as the method implemented.

### Validation methodology

`tests/unit/advanced/geometry/test_hessian.py` uses two quadratic loss
functions with an exactly-known, constant Hessian (`H = diag([2,4,6])`
and a coupled `[[2,1],[1,2]]` case): for a quadratic, `estimate_
hessian_vector_product`'s output should match `H @ v` closely regardless
of where `theta` is evaluated, which both tests confirm at `abs=1e-3`
with default step sizes, along with a direct "same `Hv` at two different
`theta`" check (a quadratic's Hessian is constant, so this must hold).

### Benchmark results

Not yet run against a real quantum circuit's loss landscape (only the
closed-form quadratic reference above, by design -- ground truth is
otherwise unavailable to check against).

### Known limitations

- **Expensive: `4 * n_parameters` loss evaluations per call.** Documented
  in the module docstring as strictly research/diagnostic-only, never
  appropriate in the per-step monitoring path (plan.md §26). Not yet
  measured empirically (no entry in `benchmarks/run_overhead_benchmark.py`
  for this function specifically).
- **Cancellation error compounds** from nesting two finite differences;
  default step sizes (`eps=grad_eps=1e-4`) are a reasonable default for
  `float64` losses of order `O(1)` but not validated across a range of
  loss magnitudes/noise levels.
- **Assumes a deterministic/low-noise `loss_fn`.** A finite-shots-sampled
  loss will produce a very noisy HVP estimate; this module has no way to
  detect or warn about that at call time.

---

## Issue #87 -- Loss-landscape sampling (`loss_landscape.py`)

### Mathematical description

`sample_loss_landscape_1d`/`_2d` evaluate `loss_fn(parameters + alpha *
direction [+ beta * direction2])` over an evenly spaced grid of
coefficients, returning the raw sampled values (no fitting/smoothing).
`random_direction` draws an isotropic random unit vector (Gaussian
components, then normalized) for scanning along an arbitrary direction
when no specific direction (e.g. a QFIM eigenvector) is of interest.
`landscape_flatness` reports plain descriptive statistics (`range`,
`std`, `mean`) of a sampled landscape -- a summary, not a verdict, per
the blueprint's detection/diagnosis separation applied here.

### References

Li, H. et al. (2018), *Visualizing the Loss Landscape of Neural Nets* --
the general 1D/2D random-direction loss-landscape-scan technique this
module adapts (without that paper's filter-normalization step, which is
specific to convolutional-network weight scaling and not applicable to
quantum-circuit parameters; see Known Limitations).

### Validation methodology

`tests/unit/advanced/geometry/test_loss_landscape.py` uses a known convex
bowl (`sum(theta**2)`) and a known flat loss as reference functions: the
bowl's sampled minimum must land at the scan's center (both for 1D and
2D grids), and the flat loss must produce exactly zero `range`/`std` via
`landscape_flatness`. Shape/endpoint-inclusion checks confirm
`n_points`/`span` are honored exactly (e.g. `alphas[0] ==
span[0]`, `alphas[-1] == span[1]`), and a `nan`-loss case confirms NaN
propagates into every summary statistic rather than being silently
masked (per addendum §7's convention for meaningful-but-abnormal
signals).

### Benchmark results

Not applicable in the Milestone-7 sense (no detection threshold to
calibrate); qualitative use against a real barren-plateau ansatz (e.g.
confirming a landscape scan around a point the QFIM/gradient flag as
suspicious does in fact look flat) is future work -- see Known
Limitations.

### Known limitations

- **No filter/scale normalization.** Unlike Li et al. (2018)'s neural-
  network-specific direction normalization (which rescales each
  direction per-layer to account for weight-scale invariance), directions
  here are used as given (e.g. raw unit vectors from `random_direction`).
  This is appropriate for quantum-circuit parameters (rotation angles, no
  analogous per-layer scale-invariance issue) but is a real difference
  from the cited technique worth stating explicitly rather than implying
  full equivalence.
- **Not yet combined end-to-end with `estimate_qfim`/
  `detect_redundant_parameters`** in an example or benchmark (e.g.
  "scan along a flagged redundant direction and confirm the loss is flat
  along it") -- each of the five Issue #83-#87 functions has been
  validated independently against closed-form references, but their
  combined behavior on a single real example circuit has not yet been
  demonstrated. This is the most valuable near-term follow-up validation
  work for the milestone.

---

## Issues #88-#89 -- Qubit- and depth-scaling analysis (`scaling.py`)

### Mathematical description

Barren-plateau theory predicts gradient-component variance shrinking
exponentially in qubit count, `Var ~ b * exp(-a * n)`, for a broad class
of expressive/deep ansätze (McClean et al. 2018 and follow-up work).
`ScalingAnalyzer.analyze_qubit_scaling(runs)` fits an ordinary-least-
squares line to `log(gradient_variance)` vs. `n_qubits` across a set of
`ScalingObservation`s and reports the fitted `slope`/`intercept`/
`r_squared`, plus a `consistent_with_exponential_decay` flag (`True` iff
the slope is meaningfully negative and, for `>= 3` points, `r_squared`
clears a threshold). `analyze_depth_scaling(runs)` is the identical
regression substituting `depth` for `n_qubits`, per the same underlying
mechanism applying to circuit depth for some ansatz families (e.g.
approach to a unitary 2-design as depth grows).

`scaling_observation_from_run_summary` bridges directly from
`reporting.summary.build_run_summary()`'s existing dict shape
(`summary["circuit"]["n_qubits"]`/`["depth"]`,
`summary["gradient"]["variance"]`), so a user running several full
`QMLMonitor` sessions at different qubit counts/depths can feed their
summaries straight in without re-extracting fields by hand.

### References

McClean, J. R. et al. (2018), *Barren plateaus in quantum neural network
training landscapes* -- the foundational result this regression is
designed to check consistency with (not prove or disprove on its own;
see Known Limitations).

### Validation methodology

`tests/unit/advanced/test_scaling.py` uses synthetic runs with **exact**
`variance = b * exp(-a * n)` (or `-a * depth`) data, so the fitted
`slope`/`r_squared` have known closed-form answers (`slope == -a` exactly,
`r_squared == 1.0` exactly) rather than only qualitative checks. Also
covered: a perfectly flat-variance case (must **not** be flagged
`consistent_with_exponential_decay`, guarding against a floating-point
sign-noise failure mode caught during development -- see Known
Limitations), a growing-variance case (must not be flagged), the
`n_points < 3` `r_squared = nan` convention, duplicate-`x`-value
aggregation (`aggregate="mean"` vs. `"none"`), a zero-variance point
(must not raise or produce `inf`/`nan` in `log_variance`), and the full
`ScalingObservation`/`scaling_observation_from_run_summary` validation
surface (invalid field values, missing circuit/gradient data in a run
summary).

**Development-time finding, documented rather than silently fixed away:**
an early version of `_analyze` classified `consistent_with_exponential_
decay` from `slope < 0` alone. For a perfectly (or near-perfectly) flat
`log(gradient_variance)` series, `numpy.polyfit`'s OLS solver can return
a nonzero slope as small as `~1e-18` of *either* sign, purely from
floating-point rounding -- which would occasionally misclassify a
genuinely flat trend (zero real correlation with qubit count/depth) as
"consistent with decay." Fixed by requiring the slope be more negative
than a fixed `_SLOPE_EPSILON = 1e-9` (many orders of magnitude below any
real barren-plateau-scale slope) before it counts as "meaningfully
negative." `test_flat_variance_is_not_consistent_with_decay` pins this
regression.

### Benchmark results

Not yet run against a real family of scaled quantum circuits (e.g.
`efficient_su2` at `n=4..16` qubits via PennyLane/Qiskit) -- only the
exact-synthetic-data unit tests above, by design (closed-form ground
truth is otherwise unavailable). This is the most important benchmark
gap in the milestone: it is the one place Milestone 12's diagnostics
could be checked against the actual literature-documented barren-plateau
scaling signature on a real ansatz, not just a hand-constructed formula.

### Known limitations

- **Correlation, not causation, and not proof.** A tight negative-slope
  fit is *consistent with* exponential-decay barren-plateau behavior; it
  is not, on its own, evidence that this specific mechanism (rather than,
  e.g., compounding shot noise, or a coincidentally-small set of runs)
  produced the observed trend. Stated in the module docstring and
  repeated here deliberately.
- **Does not control for confounding structural variables.**
  `analyze_qubit_scaling` does not verify that `depth` (or ansatz
  structure generally) was held fixed across the supplied runs, and vice
  versa for `analyze_depth_scaling` -- this is the caller's
  responsibility. An unnoticed confound (e.g. depth silently increasing
  alongside qubit count in a "qubit scaling" run set) could produce a
  fit consistent with decay for the wrong reason.
- **`r_squared_threshold=0.7`/`_SLOPE_EPSILON` are placeholder
  defaults**, not empirically calibrated against a benchmark corpus of
  known-plateauing vs. known-healthy ansatz families the way Milestone 7
  detector thresholds were -- no such corpus exists yet for this
  analyzer; building one (real circuits at several qubit counts, some
  known to plateau, some known not to) is future validation work, not
  done here.
- **Not validated against a real circuit family**, per the Benchmark
  Results gap above -- only exact-synthetic data.
