"""Internal event structures used by the monitoring core.

Milestone 2 (Volume III), supporting Issues #11-#13.

`StepObservation` bundles the framework-agnostic `TrainingEvent` (Milestone 1,
`schemas/training.py`) together with whatever optional per-step snapshots
were supplied to `QMLMonitor.update()`: a `GradientSnapshot`, `CircuitMetadata`,
`OptimizerMetadata`, shot count, and raw parameters.

Milestone 1 deliberately kept `TrainingEvent` scoped to its base fields (see
that module's docstring) rather than forward-referencing schemas that didn't
exist yet at the time. `StepObservation` is where those pieces are recombined
for consumption by the rolling window (`core/state.py`), the future
statistics engine (Milestone 3), and the future diagnosis engine
(Milestone 4) -- without requiring any change to the Milestone 1 schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.gradient import GradientSnapshot
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent


@dataclass
class StepObservation:
    """Everything observed at a single `QMLMonitor.update()` call.

    Attributes:
        training_event: The core, framework-agnostic event for this step.
        gradient: Summarized gradient information for this step, if the
            caller supplied a gradient array (via `summarize_gradient`).
        circuit: Circuit metadata for this step, if supplied.
        optimizer: Optimizer metadata for this step, if supplied.
        shots: Shot count used to produce this step's measurements, if
            supplied and using a shot-based (non-analytic) execution.
        parameters: Raw parameter vector/snapshot for this step, if
            supplied. Kept as `Any` since no `ParameterSnapshot` schema
            exists yet; not validated beyond basic presence.
    """

    training_event: TrainingEvent
    gradient: GradientSnapshot | None = None
    circuit: CircuitMetadata | None = None
    optimizer: OptimizerMetadata | None = None
    shots: int | None = None
    parameters: Any | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.training_event, TrainingEvent):
            raise TypeError(
                f"training_event must be a TrainingEvent, got {type(self.training_event)!r}"
            )
        if self.gradient is not None and not isinstance(self.gradient, GradientSnapshot):
            raise TypeError(
                f"gradient must be a GradientSnapshot or None, got {type(self.gradient)!r}"
            )
        if self.circuit is not None and not isinstance(self.circuit, CircuitMetadata):
            raise TypeError(
                f"circuit must be a CircuitMetadata or None, got {type(self.circuit)!r}"
            )
        if self.optimizer is not None and not isinstance(self.optimizer, OptimizerMetadata):
            raise TypeError(
                f"optimizer must be an OptimizerMetadata or None, got {type(self.optimizer)!r}"
            )
        if self.shots is not None:
            if not isinstance(self.shots, int) or isinstance(self.shots, bool):
                raise TypeError(f"shots must be an int or None, got {type(self.shots)!r}")
            if self.shots < 0:
                raise ValueError(f"shots must be >= 0, got {self.shots}")
