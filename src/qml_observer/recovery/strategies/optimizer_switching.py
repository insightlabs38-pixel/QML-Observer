"""OptimizerSwitchingStrategy.

Milestone 13, Issue #95 ("Optimizer-switching strategy").

Applies to `UNSTABLE` and `STAGNATION` -- issue types where the optimizer
*algorithm itself* (not just its learning rate, `LearningRateAdjustment
Strategy`'s scope) is a plausible lever:

- `UNSTABLE`: an adaptive optimizer (Adam-family) can amplify instability
  by accumulating momentum through oscillations; this strategy recommends
  switching to a more conservative gradient-descent-family optimizer, or
  a perturbation-based optimizer (SPSA) if already on a conservative one.
- `STAGNATION`: a plain gradient-descent-family optimizer with no
  momentum can stall in a shallow region an adaptive optimizer would
  escape; this strategy recommends switching to an adaptive optimizer, or
  a perturbation-based one (SPSA) if already adaptive.

Deliberately excludes `POSSIBLE_BARREN_PLATEAU`: switching *which*
classical optimizer processes the gradient signal does not address a
gradient signal that is itself vanishing -- `ParameterReinitialization
Strategy` (Issue #91) and `AnsatzDepthReductionStrategy` (Issue #94)
target that mechanism instead. A natural-gradient switch, which *does*
meaningfully change how the gradient signal itself is used, is handled
separately by `NaturalGradientStrategy` (Issue #96b).
"""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext, RecoveryRecommendation, RecoveryStrategy
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

#: Optimizer-name families used to decide which direction to switch.
#: Matched case-insensitively against `OptimizerMetadata.name`.
_ADAPTIVE_OPTIMIZERS = frozenset({"adam", "adamw", "adagrad", "rmsprop", "nadam"})
_CONSERVATIVE_OPTIMIZERS = frozenset(
    {"gradientdescent", "sgd", "vanillagradientdescent", "momentumoptimizer"}
)
_PERTURBATION_OPTIMIZERS = frozenset({"spsa", "qnspsa"})

_FALLBACK_OPTIMIZER = "GradientDescent"


class OptimizerSwitchingStrategy(RecoveryStrategy):
    """Recommends switching optimizer family based on instability vs. stagnation."""

    name = "optimizer_switching"

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return diagnosis.issue in (IssueType.UNSTABLE, IssueType.STAGNATION)

    def propose(
        self, diagnosis: DiagnosisResult, context: RecoveryContext
    ) -> RecoveryRecommendation | None:
        current_name = context.optimizer.name if context.optimizer is not None else None
        family = self._classify(current_name)

        if diagnosis.issue is IssueType.UNSTABLE:
            return self._propose_for_instability(diagnosis, current_name, family)
        return self._propose_for_stagnation(diagnosis, current_name, family)

    @staticmethod
    def _classify(name: str | None) -> str | None:
        if name is None:
            return None
        key = name.lower()
        if key in _ADAPTIVE_OPTIMIZERS:
            return "adaptive"
        if key in _CONSERVATIVE_OPTIMIZERS:
            return "conservative"
        if key in _PERTURBATION_OPTIMIZERS:
            return "perturbation"
        return "unknown"

    def _propose_for_instability(
        self, diagnosis: DiagnosisResult, current_name: str | None, family: str | None
    ) -> RecoveryRecommendation:
        if family == "adaptive":
            new_optimizer = _FALLBACK_OPTIMIZER
            reason = (
                f"Current optimizer {current_name!r} is adaptive; adaptive optimizers can "
                "amplify instability by accumulating momentum through oscillations. "
                f"Switching to a conservative optimizer ({new_optimizer}) is a standard "
                "first response."
            )
        elif family == "perturbation":
            new_optimizer = _FALLBACK_OPTIMIZER
            reason = (
                f"Current optimizer {current_name!r} is already perturbation-based; "
                f"switching to a plain, conservative optimizer ({new_optimizer}) removes "
                "an additional source of stochasticity while instability is investigated."
            )
        else:
            new_optimizer = "SPSA"
            reason = (
                f"Current optimizer {current_name or 'unknown'!r} is not clearly adaptive; "
                f"switching to a perturbation-based optimizer ({new_optimizer}) can help if "
                "instability stems from an ill-conditioned or noisy landscape rather than "
                "the optimizer's own step-size behavior."
            )
        priority = min(0.75, 0.35 + 0.3 * diagnosis.confidence)
        return RecoveryRecommendation(
            strategy_name=self.name,
            description=f"Switch optimizer to {new_optimizer}.",
            priority=priority,
            parameters={"optimizer": new_optimizer},
            rationale=[
                f"Diagnosis: unstable (confidence {diagnosis.confidence:.2f}).",
                reason,
            ],
            hook_name="set_optimizer",
        )

    def _propose_for_stagnation(
        self, diagnosis: DiagnosisResult, current_name: str | None, family: str | None
    ) -> RecoveryRecommendation:
        if family == "conservative" or family is None:
            new_optimizer = "Adam"
            reason = (
                f"Current optimizer {current_name or 'unknown'!r} has no momentum term; "
                f"switching to an adaptive optimizer ({new_optimizer}) can help escape a "
                "shallow stagnant region a plain gradient-descent-family optimizer would "
                "stall in."
            )
        elif family == "adaptive":
            new_optimizer = "SPSA"
            reason = (
                f"Current optimizer {current_name!r} is already adaptive; switching to a "
                f"perturbation-based optimizer ({new_optimizer}) is a lower-priority "
                "next step if momentum alone hasn't resolved the stagnation."
            )
        else:  # already perturbation-based
            new_optimizer = "Adam"
            reason = (
                f"Current optimizer {current_name!r} is perturbation-based; switching to an "
                f"adaptive optimizer ({new_optimizer}) offers a different escape mechanism."
            )
        priority = min(0.6, 0.25 + 0.25 * diagnosis.confidence)
        return RecoveryRecommendation(
            strategy_name=self.name,
            description=f"Switch optimizer to {new_optimizer}.",
            priority=priority,
            parameters={"optimizer": new_optimizer},
            rationale=[
                f"Diagnosis: stagnation (confidence {diagnosis.confidence:.2f}).",
                reason,
            ],
            hook_name="set_optimizer",
        )
