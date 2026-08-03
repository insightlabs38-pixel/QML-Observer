"""Milestone 14, Issue #103: writing and using a third-party detector plugin.

Shows both halves of the plugin API end to end, in one process:

1. A minimal `BaseDetector` subclass, exactly as you'd write one in your
   own installable package (see `docs/development/adding_detectors.md`).
2. Registering it as a plugin -- normally done via your package's own
   `pyproject.toml` (`[project.entry-points."qml_observer.detectors"]`);
   here, since this script *is* the "package", registration is simulated
   by monkeypatching `importlib.metadata.entry_points()` so the discovery
   functions below see it exactly as they would a real installed plugin.
3. Discovering and loading it with `qml_observer.detectors.plugins`, then
   using it in a real `QMLMonitor` run, indistinguishable from a built-in
   detector.

Run with:
    python examples/generic/detector_plugin_demo.py
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

import qml_observer.detectors.plugins as plugins_module
from qml_observer import QMLMonitor
from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.detectors.plugins import (
    discover_detector_plugins,
    list_detector_plugins,
    load_detector_plugins,
)

N_STEPS = 30


class FlatlineDetector(BaseDetector):
    """A toy third-party-style detector: fires once the loss stops changing at all.

    Deliberately simpler than the built-in `StagnationDetector` -- the
    point of this example is the plugin mechanism, not a sophisticated
    detection rule.
    """

    name = "flatline"

    def __init__(self, patience: int = 5):
        self.patience = patience
        self._last_loss: float | None = None
        self._flat_streak = 0

    def update(self, event: StepObservation, state: RunState) -> None:
        loss = event.training_event.loss
        if loss is None:
            return
        if self._last_loss is not None and loss == self._last_loss:
            self._flat_streak += 1
        else:
            self._flat_streak = 0
        self._last_loss = loss

    def diagnose(self) -> DetectorResult:
        triggered = self._flat_streak >= self.patience
        return DetectorResult(
            detector_name=self.name,
            triggered=triggered,
            confidence=min(1.0, self._flat_streak / max(self.patience, 1)) if triggered else 0.0,
            evidence=[f"loss unchanged for {self._flat_streak} consecutive steps"]
            if triggered
            else [],
            recommendations=["Check whether the optimizer step is actually being applied."]
            if triggered
            else [],
        )

    def reset(self) -> None:
        self._last_loss = None
        self._flat_streak = 0


def _simulate_installed_plugin() -> None:
    """Make `FlatlineDetector` discoverable exactly as a real installed plugin would be.

    A real third-party package does this via its own `pyproject.toml`
    entry-points table -- nothing in your code needs to call
    `entry_points()` yourself. This function exists only so this single
    script can demonstrate discovery without a second installable
    package.
    """
    fake_entry_point = EntryPoint(
        name="flatline",
        value=f"{__name__}:FlatlineDetector",
        group=plugins_module.DETECTOR_ENTRY_POINT_GROUP,
    )
    plugins_module.entry_points = lambda group=None: (
        [fake_entry_point] if group == plugins_module.DETECTOR_ENTRY_POINT_GROUP else []
    )


def main() -> None:
    _simulate_installed_plugin()

    print("Registered plugins (not yet imported):")
    print(f"  {list_detector_plugins()}\n")

    print("Discovered + validated plugin classes:")
    print(f"  {discover_detector_plugins()}\n")

    # Instantiate with a custom patience, same as any other detector.
    (flatline_detector,) = load_detector_plugins(configs={"flatline": {"patience": 8}})

    monitor = QMLMonitor(detectors=[flatline_detector], policy="log")
    print(f"Run ID: {monitor.run_id}\n")

    loss = 5.0
    for step in range(N_STEPS):
        if step >= 10:
            loss = 0.42  # loss "gets stuck" from step 10 onward
        else:
            loss *= 0.9
        diagnosis = monitor.update(step=step, loss=loss)
        print(f"step={step:>2}  loss={loss: .4f}  issue={diagnosis.issue.value}")

    final = monitor.finish()
    print(f"\nFinal diagnosis: {final.issue.value} (confidence={final.confidence:.2f})")
    print(f"Evidence: {final.evidence}")
    print(
        "\nNote: the final *issue type* stays conservative "
        "('insufficient_evidence') rather than becoming e.g. 'stagnation', "
        "even though the plugin detector clearly triggered (see Evidence "
        "above) -- DiagnosisEngine's built-in scoring (diagnosis/scoring.py) "
        "only maps specific, known detector names to an IssueType. A plugin "
        "detector's evidence still surfaces, but wiring a *new* IssueType "
        "into the scoring itself is built-in-detector RFC territory "
        "(docs/development/detector_rfc_template.md), not something a "
        "plugin can do unilaterally -- see the blueprint's detection/"
        "diagnosis separation rule."
    )


if __name__ == "__main__":
    main()
