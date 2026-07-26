# Core concepts

## The pipeline

QML Observer separates **detection** from **diagnosis** on purpose (the
blueprint's second architectural rule): a detector says "gradient collapse
detected"; the diagnosis engine decides whether that evidence is more
consistent with a possible barren plateau, ordinary convergence, or
noise-dominated optimization. Never conflate the two -- a single
detector's `DetectorResult` is not a verdict, only evidence.

```
TrainingEvent → BaseDetector.update()/.diagnose() → DetectorResult (per detector)
                                                            │
                                                            ▼
                                          DiagnosisEngine.evaluate() → DiagnosisResult
                                                            │
                                                            ▼
                                            ActionPolicy.select_action() → Action
```

## `IssueType`: what the engine can report

| Issue | Meaning |
|---|---|
| `HEALTHY` | No detector has triggered; training looks unremarkable so far. |
| `CONVERGED` | Loss has settled at a low absolute value -- given explicit priority over any other simultaneously-triggered issue, since a converged run should never be reported as a plateau. |
| `POSSIBLE_BARREN_PLATEAU` | Sustained near-zero, low-variance gradients *and* a stagnant loss. Always "possible" -- this is never definitive proof (see [How to interpret alerts](#how-to-interpret-alerts)). |
| `STAGNATION` | The optimizer itself appears frozen (loss/parameters not moving, learning rate pinned), independent of gradient magnitude -- distinct from a plateau, where gradients are the root cause. |
| `NOISE_DOMINATED` | Gradient signal-to-noise is too low to trust (Milestone 9's `NoiseDetector`; not yet part of the MVP detector set). |
| `UNSTABLE` | NaN/Inf loss or gradients -- a diverging optimizer, reported as its own distinct signal rather than silently propagating into a confidence score. |
| `INSUFFICIENT_EVIDENCE` | The default when no detectors are configured, or not enough steps have run yet to say anything. |

## Policy modes

`QMLMonitor(policy=...)` / `ActionPolicy(mode=...)` accepts:

- **`"log"`** -- always records the diagnosis; never intervenes.
- **`"warn"`** (default) -- also emits a terminal/logger warning for
  non-`"info"` severity. Never stops your loop; you decide via
  `monitor.should_stop()`.
- **`"pause"`** -- currently behaves identically to `"warn"` until
  `PauseAction` ships (Milestone 13); this is a deliberate conservative
  choice, not a placeholder bug.
- **`"stop"`** -- arms `StopAction`; `monitor.should_stop()` returns `True`
  once a detector's evidence clears its trigger threshold at `"critical"`
  severity.
- **`"adaptive"`** -- like `"stop"`, plus an explicit opt-in
  (`allow_stop_on_degraded=True`) to allow stopping even on a `degraded`
  diagnosis. Without that flag it behaves exactly like `"stop"`.

The default is deliberately conservative (plan.md §7): nothing stops your
training loop unless you explicitly ask for `"stop"`/`"adaptive"`.

## `degraded` diagnoses (fail-open policy)

If a detector or statistics function raises mid-run, `QMLMonitor` never
propagates that exception into your training loop (addendum §1). Instead,
that step's `DiagnosisResult.degraded` is `True`, with a human-readable
`degraded_reason`, and the full traceback is logged (not swallowed
silently). A degraded diagnosis is never allowed to trigger `StopAction`
except under the explicit `mode="adaptive"` + `allow_stop_on_degraded=True`
opt-in described above. Always check `diagnosis.degraded` in any
CLI/report/dashboard consumer you build and flag it visibly -- don't
present a degraded diagnosis as fully trustworthy.

## How to interpret alerts

This is the single most important thing to get right when using QML
Observer, so it gets its own section.

**A `POSSIBLE_BARREN_PLATEAU` diagnosis is a probability statement about
observed training signals, not a certificate.** Concretely:

1. **Check `confidence` and `severity` together, not `issue` alone.** A
   `"critical"`-severity, high-confidence plateau diagnosis that has
   persisted for many steps (see `evidence`, which reports the exact
   persistence count against the detector's `patience`) is much stronger
   evidence than a diagnosis that just crossed the trigger threshold.
2. **Read `evidence` before acting.** It always includes the actual
   gradient norm/variance against the configured thresholds and how many
   consecutive steps the condition has held -- this tells you *how far*
   past the threshold you are, not just that you crossed it.
3. **A small gradient alone is never sufficient.** `BarrenPlateauDetector`
   requires loss stagnation *and* gradient collapse *and* persistence
   before it can trigger at all -- if loss history is too short to confirm
   stagnation, it caps its own confidence below the triggering threshold
   rather than guessing.
4. **Default thresholds are empirically calibrated, not universal
   constants.** See `research/validation.md` for exactly how
   `BarrenPlateauDetector`'s default `gradient_threshold` was chosen, and
   recalibrate (`benchmarks/run_benchmarks.py`) if your circuits operate at
   a very different loss/gradient scale than the synthetic fixtures used
   here.
5. **Distinguish plateau from stagnation from noise.** `STAGNATION` means
   the optimizer stopped moving (check your learning-rate schedule and
   optimizer state first); `POSSIBLE_BARREN_PLATEAU` means the *gradient
   signal itself* collapsed (check circuit initialization, ansatz
   expressivity, and qubit/depth scaling first); a low-SNR
   `NOISE_DOMINATED` diagnosis (Milestone 9) means "get more shots or
   average more," not "restart the circuit."
6. **Always check `degraded` first.** A degraded diagnosis's `issue` value
   may be based on incomplete information for that step -- treat it as
   "we don't currently know," not as a confirmed healthy/unhealthy state.
