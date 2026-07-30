# Validation: calibration methodology and results

**Milestone 7, Issue #55b** ("Publish calibration methodology and results
in `docs/research/validation.md`"), required by addendum §3. This page is
the authoritative, reproducible record of how the MVP detector defaults
were chosen and what evidence supports them.

## Acceptance criteria (addendum §3)

- False-positive rate on `healthy_learning`/`convergence` fixtures
  (extended here to also cover `noise_dominated`, per plan.md §15's
  broader false-positive concern): **target < 5%**.
- Detection latency on the `artificial_plateau` fixture: **report
  median/p95 steps-to-detection, no hard target for v0.3.0** -- establish a
  baseline first.

## Procedure

1. Fixtures: the five seeded generators in `tests/fixtures/synthetic_runs.py`
   (Milestone 4, Issue #32) -- deterministic, framework-agnostic, and
   already used by unit tests, so calibration results are directly
   comparable to what the test suite already validates qualitatively.
2. Detector configuration under test: the three MVP detectors
   (`BarrenPlateauDetector`, `StagnationDetector`, `ConvergenceDetector`)
   as `QMLMonitor` would run them together, with `patience=15` (chosen to
   keep the fixtures' default `n_steps=60` long enough to observe several
   post-trigger steps without needing unreasonably long synthetic runs).
3. For each candidate `gradient_threshold`, run every scenario for 50
   seeds (`seed=0..49`) through a fresh `QMLMonitor`/detector set per
   run, and record:
   - Whether `POSSIBLE_BARREN_PLATEAU` was ever reported (false positive,
     for the three "must never trigger" fixtures).
   - The step index of the first `POSSIBLE_BARREN_PLATEAU` report, if any
     (for `artificial_plateau`, the "must trigger, and how fast" fixture).
4. Pick the smallest threshold that clears both the false-positive target
   and achieves reliable detection -- see `research/benchmarks.md` for the
   full sweep table and the specific decision for each shipped default.
5. Confirm the chosen default against the live-circuit demo
   (`examples/{pennylane,qiskit}/barren_plateau_demo.py`) as an additional
   sanity check beyond the synthetic fixtures -- both must show the
   healthy scenario completing normally and the engineered-plateau
   scenario stopping early.

This exact procedure is implemented in `benchmarks/run_benchmarks.py`
(`run_calibration_sweep()`, `run_false_positive_benchmark()`,
`run_detection_latency_benchmark()`, `run_full_benchmark()`) and walked
through narratively in `benchmarks/qml_observer_benchmark.ipynb` -- run
either to reproduce every number on this page exactly.

## Current results (v0.3.0 defaults)

`gradient_threshold=5e-6`, `patience=15`, n=50 seeds/scenario:

| Fixture | Result | Target | Status |
|---|---|---|---|
| `healthy_learning` false-positive rate | 0.0% | < 5% | met |
| `convergence` false-positive rate | 0.0% | < 5% | met |
| `noise_dominated` false-positive rate | 0.0% | < 5% | met |
| `artificial_plateau` detection rate | 100.0% | baseline (no hard target) | -- |
| `artificial_plateau` median steps-to-detection | 14 | baseline (no hard target) | -- |
| `artificial_plateau` p95 steps-to-detection | 21 | baseline (no hard target) | -- |

Raw output: `benchmarks/results/calibration_results.json` (regenerate with
`python benchmarks/run_benchmarks.py --seeds 50 --json benchmarks/results/calibration_results.json`).

## Known limitations of this validation

- **Synthetic, not real circuits.** These fixtures are hand-constructed
  loss/gradient sequences, not outputs of real quantum simulators. They
  validate the *statistical logic* of the detectors, not whether a real
  15+-qubit random-circuit barren plateau produces gradients at exactly
  this scale. The live-circuit demos (Step 5 above) are the current
  cross-check against that gap; a dedicated real-circuit benchmark suite
  (beyond the two small engineered-plateau demo circuits) is future work.
- **No depth/qubit-scaling sweep yet.** plan.md §15's "depth scaling case"
  needs `ScalingAnalyzer` (Milestone 12) to generate genuinely-scaling
  circuit families; it is not covered here.
- **Single detector-configuration sweep.** Only `gradient_threshold` was
  swept for v0.3.0; `variance_threshold` (derived automatically as
  `gradient_threshold ** 2`), `loss_improvement_threshold`, and `patience`
  were left at their original values. A joint sweep across all four is
  reasonable future work if false positives/negatives are observed on
  real circuits that this synthetic suite doesn't anticipate.
- **`StagnationDetector`/`ConvergenceDetector` thresholds are still
  unswept placeholders.** This validation focused on
  `BarrenPlateauDetector` because it was the parameter with a
  demonstrated 0%-detection failure mode; the other two detectors'
  defaults have not yet shown a similar problem against these fixtures,
  but have also not been explicitly calibrated the same way. Recalibrate
  them the same way (via `run_calibration_sweep()`, extended to their
  parameters) if real-world usage surfaces false positives/negatives.
  (A related, non-threshold gap in `StagnationDetector`'s *trigger logic*
  -- not a threshold value -- was found and fixed during the pre-release
  review; see `CHANGELOG.md`'s `[0.3.0]` "Fixed" section and
  `tests/unit/detectors/test_stagnation.py::TestLossOnlyStagnation`.)

Any future change to these defaults must update this page's results table
and add an entry to `research/benchmarks.md` plus a `CHANGELOG.md` entry,
per addendum §3.

## Milestone 9: `NoiseDetector` / finite-shots calibration (Issues #64-#68)

**Acceptance criteria (addendum §3, applied to the new detector):**
- False-positive rate (misdiagnosed as `POSSIBLE_BARREN_PLATEAU`) on the
  `finite_shots_healthy` fixture: **target < 5%**.
- `finite_shots_plateau` detection rate: **report, no hard target for
  v0.3.0** -- establish a baseline, same as Issue #54.
- New Milestone-9-specific concern: the *conflation* rate -- how often a
  genuine plateau is reported as `NOISE_DOMINATED` only (never also
  `POSSIBLE_BARREN_PLATEAU`) -- should be as close to 0% as practical,
  per Issue #67's requirement that the two never be conflated.

**Procedure:** `run_noise_benchmark()` (`benchmarks/run_benchmarks.py`)
runs the full Milestone 9 detector set (`BarrenPlateauDetector`,
`StagnationDetector`, `ConvergenceDetector`, `NoiseDetector`, all at
`patience=15`, `NoiseDetector.snr_threshold=1.0`) against
`finite_shots_healthy_run`/`finite_shots_plateau_run`
(`tests/fixtures/synthetic_runs.py`, Issue #68) across a sweep of shot
budgets, 50 seeds each.

**Results** (`snr_threshold=1.0`, `patience=15`, n=50 seeds/scenario/shot-budget):

| Shots | `finite_shots_healthy` false-positive (plateau) | `finite_shots_healthy` flagged `NOISE_DOMINATED` | `finite_shots_plateau` detection rate | `finite_shots_plateau` conflated as noise-only |
|---|---|---|---|---|
| 1 | 0.0% | 90.0% | 98.0% | 2.0% |
| 5 | 0.0% | 0.0% | 100.0% | 0.0% |
| 20 | 0.0% | 0.0% | 100.0% | 0.0% |
| 100 | 0.0% | 0.0% | 100.0% | 0.0% |
| 1000 | 0.0% | 0.0% | 100.0% | 0.0% |

Raw output: `benchmarks/results/calibration_results.json`
(`noise_shot_budget` key; regenerate with
`python benchmarks/run_benchmarks.py --seeds 50 --json benchmarks/results/calibration_results.json`).

**Findings:**
- Across the entire swept range, `NoiseDetector`'s default
  `snr_threshold=1.0` never produces a false-positive `POSSIBLE_BARREN_PLATEAU`
  reading on the healthy fixture, and never prevents `BarrenPlateauDetector`
  from detecting the genuinely-collapsed fixture -- the two detectors stay
  well-separated at every shot budget tested.
- At `shots=1` (a single measurement per expectation value -- an extreme,
  almost certainly unrealistic budget in practice), 90% of otherwise-healthy
  runs are correctly flagged `NOISE_DOMINATED` (the gradient estimate
  genuinely is untrustworthy at that budget), and in 2% of seeds on the
  *plateau* fixture the Issue #67 priority rule still resolves the headline
  diagnosis to `NOISE_DOMINATED` alone rather than `POSSIBLE_BARREN_PLATEAU`
  -- i.e. even with the priority override, an extremely small shot budget
  can occasionally suppress the plateau reading for a step or two before
  `BarrenPlateauDetector`'s own persistence window catches up. This is
  noted as a known limitation below rather than hidden.
- At `shots=5` and above, no conflation is observed in this benchmark at
  all: the default `snr_threshold=1.0` is a reasonable initial value for
  any shot budget realistic for near-term devices/simulators. No change to
  the placeholder default is made based on this data alone.

## Milestone 9: Issue #69b -- reconciliation check

Addendum §3 warns that adding a new detector/signal to the shared scoring
function can shift false-positive rates on fixtures that predate it, not
just introduce a new issue type. `run_reconciliation_check()`
(`benchmarks/run_benchmarks.py`) checks this directly: it re-runs the
Milestone 7 false-positive/detection-latency benchmarks with and without
`NoiseDetector` in the detector set, against the same seeds. Result: all
four numbers (`healthy_learning`/`convergence`/`noise_dominated`
false-positive rates, `artificial_plateau` detection rate/median/p95
latency) are identical either way -- expected, since none of those
fixtures report a `shots` field and `NoiseDetector` abstains without one,
but now a checked fact rather than an assumption. Full numbers and
rationale in `docs/research/benchmarks.md`'s Issue #69b entry.

## Known limitations of the Milestone 9 validation

- **Extreme shot budgets are undertested.** `shots=1` is included mainly
to stress-test the `NoiseDetector`/`BarrenPlateauDetector` boundary, not
because it's a realistic configuration; the 2% conflation rate observed
there should not be read as "2% of real runs will be misdiagnosed" --
no realistic shot budget this small was found to conflate at all.
- **Single detector-configuration sweep.** As with the v0.3.0 sweep, only
  `shots` was varied; `NoiseDetector.snr_threshold` and `patience` were
  held fixed. A joint sweep is reasonable future work if real usage
  surfaces false positives/negatives at shot budgets or thresholds outside
  what was tested here.
- **Synthetic fixtures, not real shot noise.** `finite_shots_healthy_run`/
  `finite_shots_plateau_run` attach a `shots` value to gradient arrays that
  are *not* actually resampled at that shot count -- they reuse the same
  underlying Gaussian-generated gradients as the analytic fixtures. This
  validates the *statistical logic* connecting `shots` to
  `NoiseDetector`'s confidence, not whether a real finite-shots PennyLane/
  Qiskit execution produces gradients matching this exact noise model. See
  `examples/pennylane/noisy_training.py` for the corresponding live-circuit
  demonstration.
