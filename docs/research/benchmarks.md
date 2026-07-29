# Benchmarks log

This page is the versioned record of default-threshold changes, per
addendum §3: "Re-calibration is a documented, versioned event." Every
entry corresponds to a `CHANGELOG.md` entry.

## How to reproduce any entry below

```bash
python benchmarks/run_benchmarks.py --seeds 50 --json benchmarks/results/calibration_results.json
```

or, for a specific parameter sweep:

```python
from benchmarks.run_benchmarks import run_calibration_sweep

run_calibration_sweep([1e-8, 1e-7, 1e-6, 2e-6, 5e-6, 1e-5], n_seeds=50)
```

## v0.2.0 -- `BarrenPlateauDetector.gradient_threshold`: `1e-8` → `5e-6`

**Why:** the blueprint's original `1e-8` placeholder achieved a 0% false
positive rate but **also a 0% detection rate** on the `artificial_plateau`
fixture -- that fixture's collapsed-gradient scale (`~1e-6` per
component, `~3e-6` L2 norm for 10 parameters) never fell below `1e-8` in
the first place, so `BarrenPlateauDetector` could never trigger on it at
all as shipped.

**Sweep** (`patience=15`, n=50 seeds/scenario):

| `gradient_threshold` | worst false-positive rate | plateau detection rate | median steps-to-detect | p95 steps-to-detect |
|---|---|---|---|---|
| `1e-8` | 0.0% | 0.0% | n/a | n/a |
| `1e-7` | 0.0% | 0.0% | n/a | n/a |
| `1e-6` | 0.0% | 0.0% | n/a | n/a |
| `2e-6` | 0.0% | 0.0% | n/a | n/a |
| **`5e-6`** | **0.0%** | **100.0%** | **14** | **21** |
| `1e-5` | 0.0% | 100.0% | 14 | 14 |

**Decision:** `5e-6` -- the first candidate that clears both acceptance
criteria (< 5% false positives, and actually detects the fixture it's
meant to detect), without over-widening the threshold further than
necessary (`1e-5` gives no additional benefit at this fixture scale and
would only reduce future headroom against genuinely-small-but-healthy
gradients). See `research/validation.md` for the full methodology writeup
and current acceptance-criteria status.

**Also changed:** `variance_threshold`'s default is derived as
`gradient_threshold ** 2`, so it moved from `1e-16` to `2.5e-11`
automatically -- no separate sweep was needed for it, since it's defined
relative to `gradient_threshold` by construction
(`detectors/barren_plateau.py`).

## Milestone 9 -- Issue #69b: reconciliation check (no threshold change)

**Why:** addendum §3 explicitly flags the risk that "adding a new signal
to the same deterministic scoring function can shift false-positive rates
on the existing fixtures, not just add a new issue type" -- Issues
#64-#69 added `NoiseDetector` and gave `NOISE_DOMINATED` priority over
`POSSIBLE_BARREN_PLATEAU` in `diagnosis/scoring.py` (Issue #67), which is
exactly the kind of scoring-function change this addendum clause warns
about. This had never been checked against the original Milestone 7
fixtures.

**Check performed:** `run_reconciliation_check()`
(`benchmarks/run_benchmarks.py`) re-runs
`run_false_positive_benchmark()`/`run_detection_latency_benchmark()`
against `healthy_learning`/`convergence`/`noise_dominated`/
`artificial_plateau` twice, with identical seeds/patience -- once with
the pre-Milestone-9 three-detector set, once with the current
four-detector set that includes `NoiseDetector` -- and diffs every
reported number.

**Result (n=50 seeds, patience=15):** all four numbers are bit-for-bit
identical before and after adding `NoiseDetector`:

| Fixture | False-positive rate, before | after |
|---|---|---|
| `healthy_learning` | 0.0% | 0.0% |
| `convergence` | 0.0% | 0.0% |
| `noise_dominated` | 0.0% | 0.0% |

| `artificial_plateau` | before | after |
|---|---|---|
| detection rate | 100.0% | 100.0% |
| median steps-to-detection | 14.0 | 14.0 |
| p95 steps-to-detection | 21 | 21 |

**Why:** none of these four fixtures attach a `shots` field, and
`NoiseDetector` abstains entirely on any step without shot-count
information (`detectors/noise.py`) -- by construction, it cannot change
the diagnosis engine's output on a purely-analytic run. This was the
expected outcome given that design, but per the blueprint's own closing
rule ("make the MVP scientifically falsifiable... explicitly benchmark
false positives and false negatives rather than assuming"), it is now a
checked fact rather than an assumption, and reproducible via
`run_reconciliation_check()`/`benchmarks/results/calibration_results.json`
(`reconciliation_check` key).

**Decision:** no threshold changes. Nothing in Milestones 7 or 9 needs
recalibration as a result of this check.

## Milestone 9 -- `NoiseDetector` shipped, `snr_threshold`: placeholder `1.0` (kept)

**Why:** Issue #66 introduced `NoiseDetector` with a placeholder
`snr_threshold=1.0` (addendum §3, same "placeholder, not final" status as
every v0.2.0 detector default). Issue #68 requires the finite-shots
fixtures needed to actually evaluate it against real shot-budget
scenarios before deciding whether to change it.

**Sweep** (`patience=15`, `snr_threshold=1.0`, n=50 seeds/scenario/shot-budget,
via `run_noise_benchmark()`):

| shots | healthy false-positive (plateau) | healthy flagged `NOISE_DOMINATED` | plateau detection rate | plateau conflated as noise-only |
|---|---|---|---|---|
| 1 | 0.0% | 90.0% | 98.0% | 2.0% |
| 5 | 0.0% | 0.0% | 100.0% | 0.0% |
| 20 | 0.0% | 0.0% | 100.0% | 0.0% |
| 100 | 0.0% | 0.0% | 100.0% | 0.0% |
| 1000 | 0.0% | 0.0% | 100.0% | 0.0% |

**Decision:** keep `snr_threshold=1.0` -- across every realistic shot
budget tested (`shots >= 5`), it produces zero false positives on the
healthy fixture and zero conflation with genuine plateau detection. The
only conflation observed was at `shots=1`, an extreme budget included
specifically to stress-test the boundary rather than represent a
realistic configuration; see `research/validation.md`'s "Known
limitations" section for the full caveat. No evidence in this sweep
supports changing the placeholder, so it ships as-is pending real-circuit
validation.
