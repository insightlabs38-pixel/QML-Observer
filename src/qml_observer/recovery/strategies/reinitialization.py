"""ParameterReinitializationStrategy.

Milestone 13, Issue #91 ("Parameter re-initialization strategy").

Reinitializing parameters is the blueprint's first-listed `RecoveryPlanner`
candidate (Volume XIV) and is generally applicable whenever training
appears stuck due to something about the *starting point* rather than the
optimizer trajectory itself: a suspected barren plateau (Volume II-1's
classic bad-initialization story) or full stagnation (an optimizer that
looks "effectively frozen", blueprint Volume VI-2).

Where `CircuitMetadata.initialization` is known and looks like a generic
default (e.g. `"random_uniform"`), this strategy specifically recommends a
reduced-domain/small-angle initialization -- a documented mitigation for
gradient-vanishing barren plateaus in the literature (small initial
rotation angles keep the circuit closer to identity, away from the
concentration-of-measure regime that drives gradients to zero for generic
deep circuits). Per this strategy's own Definition-of-Done gate (blueprint
Volume XVIII, "for research features additionally require: mathematical
description, references..."), see `docs/architecture/recovery.md` for the
specific references and known limitations of this heuristic -- it is not
re-derived or re-justified here.
"""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext, RecoveryRecommendation, RecoveryStrategy
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

#: Issue types for which reinitializing parameters is a meaningful remedy:
#: both are "training looks stuck" verdicts where the current parameter
#: trajectory itself is the suspect, as opposed to e.g. NOISE_DOMINATED
#: (a measurement-budget problem) or UNSTABLE (a step-size problem).
_APPLICABLE_ISSUES = frozenset({IssueType.POSSIBLE_BARREN_PLATEAU, IssueType.STAGNATION})

#: Initialization names considered "generic" -- i.e. not already a
#: barren-plateau-aware small-angle strategy, and therefore worth
#: recommending a change *away* from.
_GENERIC_INITIALIZATIONS = frozenset({"random_uniform", "uniform", "random", "normal", "zeros"})

_SUGGESTED_INITIALIZATION = "reduced_domain"


class ParameterReinitializationStrategy(RecoveryStrategy):
    """Recommends reinitializing circuit parameters, optionally with a safer strategy."""

    name = "parameter_reinitialization"

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return diagnosis.issue in _APPLICABLE_ISSUES

    def propose(
        self, diagnosis: DiagnosisResult, context: RecoveryContext
    ) -> RecoveryRecommendation | None:
        current_init = context.circuit.initialization if context.circuit is not None else None
        looks_generic = current_init is None or current_init.lower() in _GENERIC_INITIALIZATIONS

        rationale = [
            f"Diagnosis: {diagnosis.issue.value} (confidence {diagnosis.confidence:.2f}).",
        ]

        if diagnosis.issue is IssueType.POSSIBLE_BARREN_PLATEAU and looks_generic:
            # Barren plateau + a generic initialization is the textbook case
            # this strategy exists for: propose the well-known mitigation.
            init_desc = repr(current_init) if current_init else "unknown (assumed generic)"
            rationale.append(
                f"Circuit initialization is {init_desc}; a reduced-domain/small-angle "
                "initialization is a documented mitigation for gradient-vanishing "
                "barren plateaus."
            )
            priority = min(1.0, 0.55 + 0.35 * diagnosis.confidence)
            parameters = {"initialization": _SUGGESTED_INITIALIZATION}
            description = (
                "Reinitialize parameters using a reduced-domain/small-angle "
                f"strategy (currently {current_init!r})."
            )
        elif diagnosis.issue is IssueType.POSSIBLE_BARREN_PLATEAU:
            # Already using a non-generic initialization: still worth a
            # reinit (a different random draw from the same strategy can
            # matter), but with lower priority since the likelier culprit
            # is elsewhere (depth/entanglement -- out of this strategy's
            # scope, see Milestone 13's not-yet-implemented depth-reduction
            # strategy, Issue #94).
            rationale.append(
                f"Circuit already uses a non-generic initialization "
                f"({current_init!r}); a plain reinitialization is a lower-priority "
                "candidate here than for a generic-initialization run."
            )
            priority = min(0.5, 0.2 + 0.2 * diagnosis.confidence)
            parameters = {}
            description = (
                f"Reinitialize parameters (current strategy {current_init!r} already "
                "barren-plateau-aware; consider circuit depth/entanglement instead)."
            )
        else:
            # STAGNATION: the optimizer/trajectory looks frozen rather than
            # the gradient looking vanishingly small -- reinit is a
            # reasonable but moderate-priority "escape a bad local
            # trajectory" suggestion, independent of initialization style.
            rationale.append(
                "Training appears stagnant (loss/parameters effectively frozen); "
                "reinitializing can help escape a poor local trajectory."
            )
            priority = min(0.6, 0.3 + 0.25 * diagnosis.confidence)
            parameters = {}
            description = "Reinitialize parameters to escape the current stagnant trajectory."

        return RecoveryRecommendation(
            strategy_name=self.name,
            description=description,
            priority=priority,
            parameters=parameters,
            rationale=rationale,
            hook_name="reinitialize_parameters",
        )
