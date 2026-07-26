# `StagnationDetector`

**Module:** `qml_observer.detectors.stagnation`
**Reports:** `IssueType.STAGNATION`

Detects that the **optimizer itself** has effectively frozen -- loss not
changing, parameters not changing, learning rate pinned near zero --
independent of gradient magnitude. This is the key distinction from
`BarrenPlateauDetector`: a plateau's root cause is the *gradient signal*
collapsing; stagnation's root cause is the optimizer not applying whatever
gradient it has. A run can have large, informative gradients and still be
diagnosed as stagnant if the optimizer state itself is frozen (e.g.
`learning_rate=0.0`).

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `loss_threshold` | `1e-6` | Relative loss-improvement magnitude below which the loss is considered stagnant over the current window. |
| `patience` | `100` | Size of the rolling window (in steps) over which stagnation is assessed. |

## What to check when this triggers

Check your learning-rate schedule, optimizer state, and whether gradients
are actually reaching the optimizer at all -- not your circuit's
initialization or expressivity (that's the barren-plateau checklist).
