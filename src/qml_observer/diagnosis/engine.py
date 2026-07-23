"""DiagnosisEngine: drives detectors and combines their output each step.

Milestone 4 (Volume VII), Issue #29.

Per the blueprint's second architectural rule, detector outputs
(`DetectorResult`, one per detector, Volume V) must never be exposed
directly as the final diagnosis. `DiagnosisEngine.evaluate()` is the one
seam that turns "N independent per-detector opinions" into "one
explainable `DiagnosisResult`" (`schemas.diagnosis.DiagnosisResult`), by
delegating the actual combination logic to
`diagnosis.scoring.combine_detector_results` (Issue #30). This module is
responsible only for *driving* the detectors (calling `update()` then
`diagnose()` on each, once per step) and passing their results through;
see `diagnosis.scoring` for how those results are turned into one verdict.
"""

from __future__ import annotations

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.diagnosis.scoring import combine_detector_results
from qml_observer.schemas.diagnosis import DiagnosisResult


class DiagnosisEngine:
    """Drives a list of detectors and combines their output each step."""

    def __init__(self, detectors: list[BaseDetector], weights: dict[str, float] | None = None):
        """Create an engine over the given detectors.

        Args:
            detectors: Detectors to run, in no particular required order.
                An empty list is allowed; `evaluate()` will always return
                `INSUFFICIENT_EVIDENCE` in that case.
            weights: Optional per-detector-name confidence weights, passed
                through to `combine_detector_results` on every `evaluate()`
                call. See `diagnosis.scoring` for semantics.

        Raises:
            TypeError: If `detectors` is not a list of `BaseDetector`.
        """
        if not isinstance(detectors, list):
            raise TypeError(f"detectors must be a list, got {type(detectors)!r}")
        for i, d in enumerate(detectors):
            if not isinstance(d, BaseDetector):
                raise TypeError(f"detectors[{i}] must be a BaseDetector, got {type(d)!r}")
        self._detectors = list(detectors)
        self._weights = dict(weights) if weights else {}

    @property
    def detectors(self) -> list[BaseDetector]:
        """The detectors this engine drives (read-only view)."""
        return list(self._detectors)

    def evaluate(self, event: StepObservation, state: RunState) -> DiagnosisResult:
        """Feed `event` to every detector and combine their verdicts.

        Args:
            event: The `StepObservation` just recorded by the monitor.
            state: The run's overall rolling state, passed through to
                each detector's `update()`.

        Returns:
            A single `DiagnosisResult` combining all detector output for
            the current step. Never raises on a well-formed `event`;
            callers relying on the fail-open policy (addendum §1) should
            still wrap `evaluate()` per that policy, since individual
            detector implementations are not guaranteed exception-free.
        """
        results: list[DetectorResult] = []
        for detector in self._detectors:
            detector.update(event, state)
            results.append(detector.diagnose())
        return combine_detector_results(results, self._weights)

    def reset(self) -> None:
        """Reset every driven detector back to its construction-time state."""
        for detector in self._detectors:
            detector.reset()
