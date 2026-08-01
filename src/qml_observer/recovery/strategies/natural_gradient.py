"""NaturalGradientStrategy.

Milestone 13, Issue #96b ("Natural-gradient method as a recovery
strategy" -- explicitly named in plan.md §22 and the blueprint's
`RecoveryPlanner` candidate list, Volume XIV, but missing from the
original #90-#97 issue breakdown; added per `future_milestones_plan.md`'s
gap analysis, alongside Issue #86's Hessian-vector-product work from
Milestone 12).

Applies to `POSSIBLE_BARREN_PLATEAU` and `STAGNATION`. The quantum
natural gradient preconditions the ordinary gradient by the inverse
quantum Fisher information matrix (QFIM, `qml_observer.advanced.
geometry.qfim`, Milestone 12), which accounts for the circuit's actual
information geometry rather than treating the parameter space as flat
Euclidean space. This is a more invasive and more expensive change than
`LearningRateAdjustmentStrategy`'s step-size tweak or `OptimizerSwitching
Strategy`'s classical-optimizer swap, so it is scored at a distinctly
lower priority than those two -- see priority rationale in `propose()`.
"""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext, RecoveryRecommendation, RecoveryStrategy
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

_APPLICABLE_ISSUES = frozenset({IssueType.POSSIBLE_BARREN_PLATEAU, IssueType.STAGNATION})

#: Optimizer names already considered natural-gradient-aware; recommending
#: a switch when already on one of these would be a no-op.
_ALREADY_NATURAL_GRADIENT = frozenset({"qnspsa", "quantumnaturalgradient", "qng"})

_SUGGESTED_OPTIMIZER = "QuantumNaturalGradient"


class NaturalGradientStrategy(RecoveryStrategy):
    """Recommends switching to a quantum-natural-gradient-aware optimizer."""

    name = "natural_gradient"

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return diagnosis.issue in _APPLICABLE_ISSUES

    def propose(
        self, diagnosis: DiagnosisResult, context: RecoveryContext
    ) -> RecoveryRecommendation | None:
        current_name = context.optimizer.name if context.optimizer is not None else None
        if current_name is not None and current_name.lower() in _ALREADY_NATURAL_GRADIENT:
            # Already using a natural-gradient-aware optimizer: nothing
            # new to propose here. Let other strategies (depth reduction,
            # reinitialization) carry the recommendation set instead.
            return None

        n_qubits = context.circuit.n_qubits if context.circuit is not None else None
        rationale = [
            f"Diagnosis: {diagnosis.issue.value} (confidence {diagnosis.confidence:.2f}).",
            "The quantum natural gradient preconditions the ordinary gradient by the "
            "inverse quantum Fisher information matrix, accounting for the circuit's "
            "actual information geometry rather than treating parameter space as flat.",
        ]
        if n_qubits is not None:
            rationale.append(
                f"Note: QFIM estimation cost grows with circuit size (n_qubits={n_qubits}); "
                "see qml_observer.advanced.geometry for the cost/usage guidance this "
                "recommendation relies on before adopting it for a large circuit."
            )
        else:
            rationale.append(
                "Note: QFIM estimation cost grows with circuit size; see "
                "qml_observer.advanced.geometry for cost/usage guidance before adopting "
                "this for a large circuit."
            )

        # Deliberately capped well below LearningRateAdjustmentStrategy's
        # and OptimizerSwitchingStrategy's priorities: this is a more
        # invasive, more computationally expensive change (QFIM estimation
        # per step) that should be reached for after cheaper interventions
        # have been considered, not as a first response.
        priority = min(0.4, 0.15 + 0.2 * diagnosis.confidence)

        return RecoveryRecommendation(
            strategy_name=self.name,
            description=(
                f"Switch to a quantum-natural-gradient-aware optimizer ({_SUGGESTED_OPTIMIZER})."
            ),
            priority=priority,
            parameters={"optimizer": _SUGGESTED_OPTIMIZER},
            rationale=rationale,
            hook_name="set_optimizer",
        )
