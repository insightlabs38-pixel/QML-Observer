"""Detector interface shared by all training-pathology detectors.

Milestone 4 (Volume V), Issue #25 ("Implement BaseDetector").

Every concrete detector (`BarrenPlateauDetector`, `StagnationDetector`,
`ConvergenceDetector`, ...) implements this same three-method contract so
the `DiagnosisEngine` (Issue #29) can drive an arbitrary list of them
uniformly, and so third-party detectors (Milestone 14) can plug into the
same engine without special-casing.

Per the blueprint's second architectural rule (detection vs. diagnosis is
one of the project's two most important rules -- see blueprint's "Final
Technical Recommendation"): a detector's job stops at, e.g., "gradient
collapse detected, confidence 0.8". It never itself decides whether that
means a barren plateau, healthy convergence, or noise -- that
interpretation belongs to the `DiagnosisEngine` alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.schemas._validation import check_range, check_str_list, check_type


@dataclass
class DetectorResult:
    """A single detector's raw output for the current state of a run.

    Attributes:
        detector_name: Stable identifier for the detector that produced
            this result (e.g. "barren_plateau"), used by the
            `DiagnosisEngine` and reporting layer to attribute evidence.
        triggered: Whether this detector's condition currently holds.
            `False` does not necessarily mean "healthy" -- it may simply
            mean this detector has insufficient evidence yet; check
            `evidence` for that distinction.
        confidence: Confidence in `triggered`, in `[0, 1]`. Detectors may
            report a nonzero confidence even when `triggered` is `False`
            (e.g. "trending toward plateau but persistence not yet met"),
            which the `DiagnosisEngine` can use to distinguish a
            near-miss from a genuinely healthy run.
        evidence: Human-readable evidence strings backing this result
            (e.g. "gradient norm 2.4e-9 for 240 consecutive steps").
        recommendations: Human-readable suggested next steps, specific to
            this detector's concern. May be empty when not `triggered`.
    """

    detector_name: str
    triggered: bool
    confidence: float
    evidence: list[str]
    recommendations: list[str]

    def __post_init__(self) -> None:
        check_type(self.detector_name, str, "detector_name")
        if not self.detector_name.strip():
            raise ValueError("detector_name must be a non-empty string")
        check_type(self.triggered, bool, "triggered")
        check_range(self.confidence, 0.0, 1.0, "confidence")
        check_str_list(self.evidence, "evidence")
        check_str_list(self.recommendations, "recommendations")


class BaseDetector(ABC):
    """Abstract interface every concrete detector must implement.

    Detectors are stateful: `update()` is called once per training step
    (via `DiagnosisEngine.evaluate`) to feed the detector a new
    observation, and `diagnose()` may be called any number of times
    afterward to read the detector's current verdict without mutating
    state. `reset()` clears all internal state back to construction-time
    defaults.

    Concrete detectors own their own internal history (typically one or
    more `qml_observer.statistics.RollingWindow`s sized to their own
    `patience`/window parameter) -- this base class specifies no storage
    of its own, and neither `update()` nor `diagnose()` are expected to
    mutate `state` (the shared `RunState`); it is read-only context.

    Not thread-safe, consistent with the rest of the core/statistics
    layers in v0.1 (addendum, Concurrency / Distributed Training).
    """

    #: Stable identifier for this detector, used in
    #: `DetectorResult.detector_name` and by config/reporting to refer to
    #: this detector by name. Concrete subclasses must override this.
    name: str = "base"

    @abstractmethod
    def update(self, event: StepObservation, state: RunState) -> None:
        """Incorporate a newly observed training step.

        Args:
            event: The `StepObservation` just recorded by the monitor
                (typically `state.latest_observation`, passed explicitly
                so detectors don't need to re-derive it from `state`).
            state: The run's overall rolling state, for detectors that
                need run-level context (e.g. `state.step_count`) beyond
                what they track in their own internal windows. Detectors
                must treat this as read-only.
        """
        raise NotImplementedError

    @abstractmethod
    def diagnose(self) -> DetectorResult:
        """Return this detector's current verdict, without mutating state.

        May be called zero or more times between `update()` calls (e.g.
        by the `DiagnosisEngine` after every step, or by a caller
        inspecting mid-run state); calling it must never itself change
        what a subsequent `diagnose()` call would return.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state, as if no `update()` had been called."""
        raise NotImplementedError
