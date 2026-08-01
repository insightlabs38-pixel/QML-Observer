"""LearningRateAdjustmentStrategy.

Milestone 13, Issue #92 ("Learning-rate adjustment strategy").

Applies to the two issue types where the *step size* (rather than the
starting point or the shot budget) is the more plausible lever:

- `IssueType.UNSTABLE` -- loss/gradients diverging or oscillating wildly
  is the textbook symptom of too large a learning rate; this strategy
  recommends halving it.
- `IssueType.STAGNATION` -- an "effectively frozen" optimizer (blueprint
  Volume VI-2) can mean the step size has decayed (or was set) too small
  to make visible progress; this strategy recommends a modest increase.

Deliberately does *not* apply to `POSSIBLE_BARREN_PLATEAU`: a vanishing
gradient signal isn't fixed by a larger learning rate (multiplying an
already-vanishing gradient by a larger step still vanishes), so recommending
a learning-rate change there would be a misleading remedy for the wrong
mechanism -- `ParameterReinitializationStrategy` (Issue #91) and the
not-yet-implemented depth-reduction/natural-gradient strategies are the
better-targeted candidates for that issue type.
"""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext, RecoveryRecommendation, RecoveryStrategy
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

#: Multiplicative adjustment applied to the current learning rate.
_UNSTABLE_DECREASE_FACTOR = 0.5
_STAGNATION_INCREASE_FACTOR = 2.0

#: Fallback learning rate proposed when the current one is unknown
#: (`context.optimizer` missing or `learning_rate=None`). A conservative,
#: commonly-used default rather than a guess tied to any specific run.
_FALLBACK_LEARNING_RATE = 0.01


class LearningRateAdjustmentStrategy(RecoveryStrategy):
    """Recommends decreasing (UNSTABLE) or increasing (STAGNATION) the learning rate."""

    name = "learning_rate_adjustment"

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return diagnosis.issue in (IssueType.UNSTABLE, IssueType.STAGNATION)

    def propose(
        self, diagnosis: DiagnosisResult, context: RecoveryContext
    ) -> RecoveryRecommendation | None:
        current_lr = context.optimizer.learning_rate if context.optimizer is not None else None

        if diagnosis.issue is IssueType.UNSTABLE:
            return self._propose_decrease(diagnosis, current_lr)
        return self._propose_increase(diagnosis, current_lr)

    def _propose_decrease(
        self, diagnosis: DiagnosisResult, current_lr: float | None
    ) -> RecoveryRecommendation:
        if current_lr is not None and current_lr > 0:
            new_lr = current_lr * _UNSTABLE_DECREASE_FACTOR
            rationale = [
                f"Diagnosis: unstable (confidence {diagnosis.confidence:.2f}).",
                f"Current learning rate {current_lr:.6g}; halving it is a standard "
                "first response to divergent/oscillating optimization.",
            ]
        else:
            new_lr = _FALLBACK_LEARNING_RATE * _UNSTABLE_DECREASE_FACTOR
            rationale = [
                f"Diagnosis: unstable (confidence {diagnosis.confidence:.2f}).",
                "Current learning rate unknown; proposing a conservative reduced "
                f"default ({new_lr:.6g}) rather than a specific halving.",
            ]
        # Instability is a strong, directly-actionable signal; priority
        # tracks diagnosis confidence closely.
        priority = min(1.0, 0.6 + 0.35 * diagnosis.confidence)
        return RecoveryRecommendation(
            strategy_name=self.name,
            description=f"Decrease learning rate to {new_lr:.6g} to reduce optimizer instability.",
            priority=priority,
            parameters={"learning_rate": new_lr},
            rationale=rationale,
            hook_name="set_learning_rate",
        )

    def _propose_increase(
        self, diagnosis: DiagnosisResult, current_lr: float | None
    ) -> RecoveryRecommendation:
        if current_lr is not None and current_lr > 0:
            new_lr = current_lr * _STAGNATION_INCREASE_FACTOR
            rationale = [
                f"Diagnosis: stagnation (confidence {diagnosis.confidence:.2f}).",
                f"Current learning rate {current_lr:.6g}; doubling it is a modest "
                "first response to an optimizer that looks effectively frozen.",
            ]
        else:
            new_lr = _FALLBACK_LEARNING_RATE
            rationale = [
                f"Diagnosis: stagnation (confidence {diagnosis.confidence:.2f}).",
                f"Current learning rate unknown; proposing a conservative default "
                f"({new_lr:.6g}) rather than a specific doubling.",
            ]
        # Stagnation is a weaker, more ambiguous signal than instability
        # (it can equally be a genuine plateau, better addressed by
        # ParameterReinitializationStrategy) -- moderate priority.
        priority = min(0.7, 0.3 + 0.3 * diagnosis.confidence)
        return RecoveryRecommendation(
            strategy_name=self.name,
            description=f"Increase learning rate to {new_lr:.6g} to escape stagnation.",
            priority=priority,
            parameters={"learning_rate": new_lr},
            rationale=rationale,
            hook_name="set_learning_rate",
        )
