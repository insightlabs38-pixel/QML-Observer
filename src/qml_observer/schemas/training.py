"""TrainingEvent schema.

A TrainingEvent represents a single observed step in a variational QML
training run. It is the core, framework-agnostic unit that all adapters
(PennyLane, Qiskit, generic, ...) convert their framework-specific
information into before it reaches the monitoring/detection layers.

Per the project blueprint (Volume II), additional structured fields
(`gradient`, `circuit`, `optimizer`, `execution`) are introduced as their
own schemas land in later Milestone 1 issues (#5-#7) and are wired into
this event at that point. Keeping this issue scoped to the base fields
avoids forward-referencing schemas that don't exist yet.
"""

from dataclasses import dataclass

from qml_observer.schemas._validation import (
    check_non_empty_str,
    check_non_negative_int,
    check_non_negative_number,
    check_type,
)


@dataclass
class TrainingEvent:
    """A single observed training step.

    Attributes:
        run_id: Identifier for the training run this event belongs to.
        step: Monotonically increasing step index within the run.
        loss: Observed loss value at this step, if available.
        epoch: Epoch index, if the training loop is epoch-based.
        timestamp: Unix timestamp (seconds) when the event was recorded.
        wall_time: Wall-clock duration (seconds) taken to produce this
            step (e.g. circuit execution + gradient computation time).
    """

    run_id: str
    step: int
    loss: float | None = None
    epoch: int | None = None
    timestamp: float | None = None
    wall_time: float | None = None

    def __post_init__(self) -> None:
        check_non_empty_str(self.run_id, "run_id")
        check_type(self.step, int, "step")
        if isinstance(self.step, bool) or self.step < 0:
            raise ValueError(f"step must be a non-negative int, got {self.step!r}")
        # loss may legitimately be NaN/Inf for a diverging optimizer (addendum §7)
        # — the detector layer classifies that, schemas must not reject it.
        if self.loss is not None:
            check_type(self.loss, (int, float), "loss")
        check_non_negative_int(self.epoch, "epoch")
        if self.timestamp is not None:
            check_type(self.timestamp, (int, float), "timestamp")
        # wall_time is a duration: negative values are never meaningful,
        # but NaN/Inf are tolerated in case of instrumentation failure.
        check_non_negative_number(self.wall_time, "wall_time")
