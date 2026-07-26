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

## v0.1.0 -- `BarrenPlateauDetector.gradient_threshold`: `1e-8` → `5e-6`

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
