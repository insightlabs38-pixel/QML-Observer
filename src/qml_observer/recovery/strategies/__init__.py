"""Concrete `RecoveryStrategy` implementations (Milestone 13).

- Issue #91: `ParameterReinitializationStrategy`
- Issue #92: `LearningRateAdjustmentStrategy`
- Issue #93: `ShotBudgetAdjustmentStrategy`
- Issue #94: `AnsatzDepthReductionStrategy`
- Issue #95: `OptimizerSwitchingStrategy`
- Issue #96b: `NaturalGradientStrategy`

The `RecoveryStrategy` interface (`recovery/base.py`) and `RecoveryPlanner`
(`recovery/planner.py`) did not need to change to accommodate any of
these later additions.
"""

from __future__ import annotations

from qml_observer.recovery.strategies.depth_reduction import AnsatzDepthReductionStrategy
from qml_observer.recovery.strategies.learning_rate import LearningRateAdjustmentStrategy
from qml_observer.recovery.strategies.natural_gradient import NaturalGradientStrategy
from qml_observer.recovery.strategies.optimizer_switching import OptimizerSwitchingStrategy
from qml_observer.recovery.strategies.reinitialization import ParameterReinitializationStrategy
from qml_observer.recovery.strategies.shot_budget import ShotBudgetAdjustmentStrategy

__all__ = [
    "ParameterReinitializationStrategy",
    "LearningRateAdjustmentStrategy",
    "ShotBudgetAdjustmentStrategy",
    "AnsatzDepthReductionStrategy",
    "OptimizerSwitchingStrategy",
    "NaturalGradientStrategy",
]
