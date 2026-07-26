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
  median/p95 steps-to-detection, no hard target for v0.1** -- establish a
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

## Current results (v0.1.0 defaults)

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
  swept for v0.1.0; `variance_threshold` (derived automatically as
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
  review; see `CHANGELOG.md`'s `[0.1.0]` "Fixed" section and
  `tests/unit/detectors/test_stagnation.py::TestLossOnlyStagnation`.)

Any future change to these defaults must update this page's results table
and add an entry to `research/benchmarks.md` plus a `CHANGELOG.md` entry,
per addendum §3.
