"""AnsatzDepthReductionStrategy.

Milestone 13, Issue #94 ("Ansatz-depth-reduction strategy").

Applies only to `IssueType.POSSIBLE_BARREN_PLATEAU`. Unlike
`ParameterReinitializationStrategy` (Issue #91), which targets the
*starting point*, this strategy targets the circuit's *expressivity*
directly: barren-plateau severity is theoretically predicted (and, per
`qml_observer.advanced.scaling.ScalingAnalyzer`, Milestone 12) to worsen
with circuit depth for sufficiently expressive, hardware-efficient
ansaetze, so reducing depth is a mechanistically direct mitigation rather
than a workaround.
"""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext, RecoveryRecommendation, RecoveryStrategy
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

#: Fraction of the current depth to remove per recommendation. A single
#: bounded reduction step, not a search over depths -- see
#: docs/architecture/recovery.md's Known Limitations for why this is a
#: fixed heuristic rather than a calibrated value.
_DEFAULT_REDUCTION_FRACTION = 0.5

#: Never propose reducing depth below this floor, so the recommendation
#: can't degenerate into an unusable near-trivial circuit.
_MIN_DEPTH = 1

#: Depth above which a barren-plateau diagnosis makes this strategy's
#: rationale unambiguous (deep circuits are exactly where the
#: exponential-concentration mechanism is expected to bite hardest).
_DEEP_CIRCUIT_THRESHOLD = 10


class AnsatzDepthReductionStrategy(RecoveryStrategy):
    """Recommends reducing circuit depth to mitigate a suspected barren plateau."""

    name = "ansatz_depth_reduction"

    def __init__(self, reduction_fraction: float = _DEFAULT_REDUCTION_FRACTION) -> None:
        """Configure the strategy.

        Args:
            reduction_fraction: Fraction of the current depth to remove,
                in `(0, 1)`. Defaults to `0.5` (halve the depth).

        Raises:
            ValueError: If `reduction_fraction` is not in `(0, 1)`.
        """
        if not (0.0 < reduction_fraction < 1.0):
            raise ValueError(f"reduction_fraction must be in (0, 1), got {reduction_fraction}")
        self._reduction_fraction = reduction_fraction

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return diagnosis.issue is IssueType.POSSIBLE_BARREN_PLATEAU

    def propose(
        self, diagnosis: DiagnosisResult, context: RecoveryContext
    ) -> RecoveryRecommendation | None:
        current_depth = context.circuit.depth if context.circuit is not None else None

        if current_depth is None or current_depth <= _MIN_DEPTH:
            # No depth to reason about, or already at the floor: this
            # strategy has nothing concrete to propose. Rather than
            # invent a depth, return None so RecoveryPlanner simply
            # excludes this strategy for this context, letting other
            # applicable strategies (e.g. reinitialization) carry the
            # recommendation set instead.
            return None

        new_depth = max(_MIN_DEPTH, round(current_depth * (1.0 - self._reduction_fraction)))
        rationale = [
            f"Diagnosis: possible barren plateau (confidence {diagnosis.confidence:.2f}).",
            f"Current circuit depth {current_depth}; barren-plateau severity is "
            "theoretically predicted to worsen with depth for expressive ansaetze, "
            f"so reducing depth to {new_depth} directly targets the mechanism "
            "(rather than only the starting point, as reinitialization does).",
        ]
        # Deeper circuits make the depth-driven mechanism a more
        # confident explanation, so weight priority up modestly for
        # circuits already past a "clearly deep" threshold.
        depth_boost = 0.1 if current_depth >= _DEEP_CIRCUIT_THRESHOLD else 0.0
        priority = min(1.0, 0.45 + 0.35 * diagnosis.confidence + depth_boost)

        return RecoveryRecommendation(
            strategy_name=self.name,
            description=f"Reduce circuit depth from {current_depth} to {new_depth}.",
            priority=priority,
            parameters={"depth": new_depth},
            rationale=rationale,
            hook_name="set_circuit_depth",
        )
