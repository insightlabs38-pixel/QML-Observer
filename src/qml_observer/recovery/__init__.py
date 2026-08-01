"""qml_observer.recovery: the recovery engine (Milestone 13, Volume XIV).

Public re-exports for `qml_observer.recovery.*`. Explicitly *not*
auto-wired into `QMLMonitor` or `ActionPolicy` (blueprint Volume XIV: "Do
not implement automatic recovery until the detection system is
validated") -- this is an opt-in layer a caller reaches for explicitly,
typically after a `PauseAction` has fired:

    from qml_observer.recovery import (
        RecoveryContext,
        RecoveryExecutor,
        RecoveryPlanner,
    )
    from qml_observer.recovery.strategies import (
        LearningRateAdjustmentStrategy,
        ParameterReinitializationStrategy,
        ShotBudgetAdjustmentStrategy,
    )

    planner = RecoveryPlanner([
        ParameterReinitializationStrategy(),
        LearningRateAdjustmentStrategy(),
        ShotBudgetAdjustmentStrategy(),
    ])
    context = RecoveryContext(run_id=monitor.run_id, step=monitor.state.step_count, ...)
    recommendations = planner.recommend(monitor.latest_diagnosis(), context)

    executor = RecoveryExecutor()
    if recommendations:
        outcome = executor.apply(recommendations[0], my_training_state)
"""

from __future__ import annotations

from qml_observer.recovery.base import (
    RecoveryContext,
    RecoveryOutcome,
    RecoveryRecommendation,
    RecoveryStrategy,
)
from qml_observer.recovery.evaluation import RecoveryEvaluationResult, RecoveryEvaluator
from qml_observer.recovery.executor import RecoveryExecutor
from qml_observer.recovery.planner import RecoveryPlanner
from qml_observer.recovery.resume import resume_monitor_from_snapshot

__all__ = [
    "RecoveryContext",
    "RecoveryRecommendation",
    "RecoveryOutcome",
    "RecoveryStrategy",
    "RecoveryPlanner",
    "RecoveryExecutor",
    "RecoveryEvaluator",
    "RecoveryEvaluationResult",
    "resume_monitor_from_snapshot",
]
