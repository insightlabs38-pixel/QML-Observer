# Adding a detector

Every detector implements `qml_observer.detectors.base.BaseDetector`:
three methods, plus a class-level `name`.

```python
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState


class MyDetector(BaseDetector):
    name = "my_detector"

    def __init__(self, *, patience: int = 50):
        self._patience = patience
        # own your own rolling state -- typically one or more
        # qml_observer.statistics.RollingWindow instances sized to patience

    def update(self, event: StepObservation, state: RunState) -> None:
        # incorporate one new observation; treat `state` as read-only
        ...

    def diagnose(self) -> DetectorResult:
        # return a verdict without mutating state -- may be called
        # any number of times between update() calls
        return DetectorResult(
            detector_name=self.name,
            triggered=...,
            confidence=...,  # in [0, 1]
            evidence=[...],  # human-readable strings
            recommendations=[...],
        )

    def reset(self) -> None:
        # clear all internal state back to construction-time defaults
        ...
```

## Rules that keep detectors composable

1. **Never decide the final diagnosis yourself.** A detector's job stops
   at "this specific condition holds, with this confidence, for these
   reasons" -- never "this is a barren plateau." That interpretation is
   the `DiagnosisEngine`'s alone (blueprint's second architectural rule).
   If your detector name-checks another detector's concern (e.g. checking
   loss stagnation *and* claiming it's specifically a plateau rather than
   stagnation), that's a sign the diagnosis-engine boundary has leaked.
2. **`triggered=False` does not mean "healthy."** It may simply mean
   insufficient evidence yet. Use `confidence` to express a graded
   "trending toward triggering but not there yet" signal so the diagnosis
   engine (and a human reading `evidence`) can distinguish a near-miss
   from genuine health.
3. **Write real evidence strings, not category labels.** Compare
   `BarrenPlateauDetector`'s evidence (`"Latest gradient norm: 2.4e-9
   (threshold 5.0e-06)."`) to a vague `"gradient collapsed"` -- the former
   lets a person or `ActionPolicy` judge how far past threshold the run
   actually is.
4. **Default thresholds are calibration targets, not guesses.** Ship a
   default, then calibrate it against `benchmarks/run_benchmarks.py`'s
   fixture suite (extend `run_calibration_sweep()` for your parameter) and
   document the result per `research/methodology.md`'s process, before
   calling the default final.
5. **Be defensive about missing data.** `event.gradient`/`event.loss` may
   be `None` (a caller didn't provide them that step) -- handle that
   without raising; the monitor's fail-open guarantee is a last resort,
   not a substitute for a detector handling its own expected-missing
   cases.
6. **Add both unit and fixture-level tests.** Unit-test `update()`/
   `diagnose()`/`reset()` directly (see `tests/unit/detectors/`), and add
   your detector to a scenario in `tests/fixtures/synthetic_runs.py` if it
   covers a genuinely new failure mode not already represented there.

## Registering with `DiagnosisEngine`

No registration step is required beyond passing an instance in
`QMLMonitor(detectors=[...])` or `DiagnosisEngine(detectors=[...])`
directly -- the engine treats every `BaseDetector` uniformly regardless of
which module it came from. See `diagnosis/scoring.py`'s
`combine_detector_results()` if your detector's evidence should interact
with another detector's (e.g. `CONVERGED`'s priority over other
simultaneously-triggered issues) rather than simply being added to the
evidence list independently.

## Third-party / plugin detectors

A documented plugin API for out-of-tree detectors shipped in Milestone 14
(`qml_observer.detectors.plugins`, Issue #103): a third-party package
registers a `BaseDetector` subclass under the `qml_observer.detectors`
entry-point group, and `load_detector_plugins()`/`discover_detector_plugins()`
find and instantiate it -- no different from importing it and passing it
to `QMLMonitor(detectors=[...])` yourself, just discoverable without
knowing the exact import path. See `docs/development/plugin_api.md` for
the full guide, and `SECURITY.md` for the (no-sandboxing) security
posture. The pattern above (implementing `BaseDetector` directly) works
identically either way -- registering an entry point is optional
packaging sugar on top of it, not a different detector interface.
