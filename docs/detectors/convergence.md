# `ConvergenceDetector`

**Module:** `qml_observer.detectors.convergence`
**Reports:** `IssueType.CONVERGED`

Distinguishes **good convergence** (loss has settled at a low absolute
value) from **bad gradient collapse** -- this distinction is essential per
the blueprint (Volume VI-3). A shrinking gradient is not itself a red flag
if the loss has genuinely reached a good optimum; only a shrinking
gradient *without* a correspondingly low loss is suspicious, and that case
is what `BarrenPlateauDetector` (not this detector) reports.

The diagnosis engine gives `CONVERGED` explicit priority over any other
simultaneously-triggered issue -- a converged run should never be reported
as a plateau just because its gradient also happens to be small at
convergence.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `loss_threshold` | `1e-3` | Absolute loss value at or below which the run is considered to have reached a "good" optimum. Unlike `BarrenPlateauDetector.loss_improvement_threshold`, this is a magnitude, not a relative-improvement figure. |
| `gradient_threshold` | `1e-4` | Supporting signal: gradient scale consistent with having settled near an optimum. |
| `patience` | `50` | Consecutive steps the converged condition must hold. |

`loss_threshold` is problem-dependent (a "good" loss for a 2-qubit toy
circuit is not a "good" loss for a real VQE Hamiltonian) -- set it to match
your problem's actual target loss scale, not this default.
