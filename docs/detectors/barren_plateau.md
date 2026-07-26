# `BarrenPlateauDetector`

**Module:** `qml_observer.detectors.barren_plateau`
**Reports:** `IssueType.POSSIBLE_BARREN_PLATEAU` (via the diagnosis engine)

## Algorithm

Straight from the blueprint (Volume VI-1):

```
IF gradient is small
AND gradient variance is small
AND loss is not improving
AND condition persists
THEN
    increase plateau confidence
```

A small gradient **alone is never sufficient** to trigger. If loss history
is too short to confirm stagnation, the detector caps its own confidence
below the triggering threshold rather than guessing -- it would rather
report "insufficient evidence" than a false plateau claim.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `gradient_threshold` | `5e-6` | Gradient L2-norm at or below which a step is "small". **Empirically calibrated** -- see `research/validation.md`. |
| `variance_threshold` | `gradient_threshold ** 2` | Gradient variance at or below which a step has a "flat" distribution, not just a small mean. |
| `loss_improvement_threshold` | `1e-6` | Relative loss-improvement magnitude below which the loss is "stagnant". |
| `patience` | `100` | Consecutive steps the small-gradient condition must hold before triggering. |

Recalibrate these for circuits operating at a very different loss/gradient
scale than the benchmark suite -- see `benchmarks/run_benchmarks.py`'s
`run_calibration_sweep()`.

## Evidence fields

Every `DetectorResult.evidence` entry reports the *actual* observed value
against its threshold and the exact persistence count, e.g.:

```
Latest gradient norm: 2.4e-9 (threshold 5.0e-06).
Latest gradient variance: 8.1e-12 (threshold 2.5e-11).
Small-gradient condition has persisted for 240 consecutive step(s) (patience 100).
Relative loss improvement over window: 1.2e-8 (stagnation threshold 1.0e-06).
```

Read these before acting -- they tell you *how far* past threshold you
are, not just that you crossed it. See
`getting_started/concepts.md#how-to-interpret-alerts`.

## What this is *not*

This detector never claims definitive proof of a barren plateau -- only
that the observed signals are consistent with one. It cannot distinguish a
genuine barren plateau from a shot-noise-starved gradient measurement
(that's `NoiseDetector`, Milestone 9) or from a bad but non-plateau
initialization that happens to start with small gradients (inspect circuit
initialization and ansatz expressivity yourself).
