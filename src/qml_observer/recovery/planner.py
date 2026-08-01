"""RecoveryPlanner: rank candidate recovery interventions for a diagnosis.

Milestone 13, Issue #90 ("Recovery strategy interface").

`RecoveryPlanner` is the recovery layer's analogue of `DiagnosisEngine`
(`diagnosis/engine.py`): given a list of `RecoveryStrategy` instances, it
asks each applicable one to `propose()` a `RecoveryRecommendation`, then
returns them ranked by `priority`, descending (plan.md §22: "rank
candidate interventions"). It never applies anything itself -- that is
`RecoveryExecutor`'s job, kept as a separate class exactly as
`ActionPolicy.select_action()` (deciding) is kept separate from an
`Action.execute()` (doing).
"""

from __future__ import annotations

import logging

from qml_observer.recovery.base import RecoveryContext, RecoveryRecommendation, RecoveryStrategy
from qml_observer.schemas.diagnosis import DiagnosisResult

_logger = logging.getLogger("qml_observer.recovery")


class RecoveryPlanner:
    """Ranks candidate recovery interventions for a `DiagnosisResult`.

    Example:
        >>> planner = RecoveryPlanner([
        ...     ParameterReinitializationStrategy(),
        ...     LearningRateAdjustmentStrategy(),
        ... ])
        >>> recommendations = planner.recommend(diagnosis, context)
        >>> best = recommendations[0] if recommendations else None
    """

    def __init__(self, strategies: list[RecoveryStrategy] | None = None) -> None:
        """Create a `RecoveryPlanner`.

        Args:
            strategies: `RecoveryStrategy` instances to consider. If
                omitted or empty, `recommend()` always returns `[]` --
                mirroring `QMLMonitor(detectors=None)`'s "no detectors
                configured" placeholder behavior rather than raising.

        Raises:
            TypeError: If any element of `strategies` is not a
                `RecoveryStrategy`.
        """
        self._strategies = list(strategies) if strategies else []
        for i, s in enumerate(self._strategies):
            if not isinstance(s, RecoveryStrategy):
                raise TypeError(f"strategies[{i}] must be a RecoveryStrategy, got {type(s)!r}")

    @property
    def strategies(self) -> list[RecoveryStrategy]:
        """The configured strategies, in the order they will be consulted."""
        return list(self._strategies)

    def recommend(
        self,
        diagnosis: DiagnosisResult,
        context: RecoveryContext,
        *,
        allow_degraded: bool = False,
    ) -> list[RecoveryRecommendation]:
        """Return recommendations for `diagnosis`, ranked by priority (descending).

        Conservative by default (addendum §1's philosophy applied to
        recovery, not just actions): a `degraded=True` diagnosis is
        evidence produced while something else was already failing, so
        recommending an intervention based on it is unreliable by
        construction. `recommend()` returns `[]` for a degraded diagnosis
        unless the caller explicitly passes `allow_degraded=True` --
        there is no `mode="adaptive"`-style escalation path here since
        `RecoveryPlanner` has no notion of "mode" at all; the caller (a
        future orchestration layer, or a human deciding whether to trust
        a degraded run's recommendations) makes that call explicitly, one
        `recommend()` call at a time.

        Each strategy's `applies_to()`/`propose()` calls are individually
        wrapped: an exception from one strategy is logged and skipped,
        never allowed to prevent other strategies from being considered
        (fail-open, addendum §1).

        Args:
            diagnosis: The diagnosis to propose recovery options for.
            context: Contextual information (circuit/optimizer/shots/
                gradient) strategies may need beyond the diagnosis itself.
            allow_degraded: Opt-in override permitting recommendations to
                be generated even for a `degraded=True` diagnosis.

        Returns:
            A list of `RecoveryRecommendation`, sorted by `priority`
            descending. Empty if no configured strategy applies, if
            every applicable strategy returned `None`, or if `diagnosis.
            degraded` is `True` and `allow_degraded` was not set.
        """
        if diagnosis.degraded and not allow_degraded:
            _logger.info(
                "qml_observer.recovery: skipping recommendations for a degraded "
                "diagnosis (run_id=%s step=%s); pass allow_degraded=True to override.",
                context.run_id,
                context.step,
            )
            return []

        recommendations: list[RecoveryRecommendation] = []
        for strategy in self._strategies:
            try:
                if not strategy.applies_to(diagnosis):
                    continue
                recommendation = strategy.propose(diagnosis, context)
            except Exception:
                _logger.warning(
                    "qml_observer.recovery: strategy %r raised while proposing a "
                    "recommendation for run_id=%s step=%s; skipping it.",
                    getattr(strategy, "name", type(strategy).__name__),
                    context.run_id,
                    context.step,
                    exc_info=True,
                )
                continue
            if recommendation is not None:
                recommendations.append(recommendation)

        recommendations.sort(key=lambda r: r.priority, reverse=True)
        return recommendations
