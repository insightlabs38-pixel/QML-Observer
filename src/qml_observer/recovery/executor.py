"""RecoveryExecutor: apply a `RecoveryRecommendation`, non-invasively.

Milestone 13, Issue #90 ("Recovery strategy interface" -- `RecoveryExecutor.
apply()` half of the blueprint's `RecoveryPlanner`/`RecoveryExecutor` pair,
Volume XIV).

Per the non-invasive core principle (plan.md §2) and the same contract
`StopAction`/`PauseAction` already use: qml_observer does not own the
caller's training loop or optimizer, so it cannot reach in and change a
learning rate or reinitialize parameters on its own. `RecoveryExecutor`
therefore applies a recommendation only via an explicit, named hook
method on a `training_state` object the *caller* supplies and controls
(see `apply()`). If that object doesn't implement the hook a
recommendation names, the executor reports the recommendation as not
applied and explains what to do manually -- it never guesses at some
other way to mutate `training_state`.
"""

from __future__ import annotations

import logging

from qml_observer.recovery.base import RecoveryOutcome, RecoveryRecommendation

_logger = logging.getLogger("qml_observer.recovery")


class RecoveryExecutor:
    """Applies a single `RecoveryRecommendation` to a caller-supplied training state.

    Example:
        >>> class MyTrainingState:
        ...     def set_learning_rate(self, learning_rate: float) -> None:
        ...         self.optimizer.lr = learning_rate
        >>> executor = RecoveryExecutor()
        >>> outcome = executor.apply(recommendation, MyTrainingState())
        >>> if outcome.applied:
        ...     print("recovery applied:", outcome.message)
    """

    def apply(
        self, recommendation: RecoveryRecommendation, training_state: object
    ) -> RecoveryOutcome:
        """Apply `recommendation` to `training_state`, if it exposes a matching hook.

        Looks up `getattr(training_state, recommendation.hook_name)`. If
        present and callable, calls it with `recommendation.parameters`
        as keyword arguments. Never raises: a hook that itself raises is
        caught and reported via `RecoveryOutcome(applied=False, ...)`
        rather than propagated, mirroring every `Action.execute()`'s
        fail-open contract (Issue #40's action-safety guarantee, applied
        here to the recovery layer).

        Args:
            recommendation: The `RecoveryRecommendation` to apply, e.g.
                from `RecoveryPlanner.recommend()`.
            training_state: A caller-supplied object that may optionally
                implement the hook named by `recommendation.hook_name`.
                qml_observer places no other requirement on this object's
                shape -- it is entirely the caller's own training-loop
                state, not a qml_observer type.

        Returns:
            A `RecoveryOutcome` describing whether the hook was found and
            ran successfully.
        """
        if recommendation.hook_name is None:
            return RecoveryOutcome(
                strategy_name=recommendation.strategy_name,
                applied=False,
                message=(
                    f"{recommendation.description} This recommendation has no "
                    "automatable hook; apply it manually."
                ),
            )

        hook = getattr(training_state, recommendation.hook_name, None)
        if hook is None or not callable(hook):
            return RecoveryOutcome(
                strategy_name=recommendation.strategy_name,
                applied=False,
                message=(
                    f"training_state does not implement "
                    f"`{recommendation.hook_name}(**{sorted(recommendation.parameters)})`; "
                    f"apply manually: {recommendation.description}"
                ),
            )

        try:
            hook(**recommendation.parameters)
        except Exception as exc:  # fail-open, same contract as Action.execute()
            _logger.warning(
                "qml_observer.recovery: applying %r via `%s` raised; "
                "the caller's training loop continues uninterrupted (fail-open policy).",
                recommendation.strategy_name,
                recommendation.hook_name,
                exc_info=True,
            )
            return RecoveryOutcome(
                strategy_name=recommendation.strategy_name,
                applied=False,
                message=(
                    f"`{recommendation.hook_name}` raised {type(exc).__name__}: {exc}. "
                    f"Apply manually: {recommendation.description}"
                ),
            )

        return RecoveryOutcome(
            strategy_name=recommendation.strategy_name,
            applied=True,
            message=f"Applied via `{recommendation.hook_name}`: {recommendation.description}",
        )
