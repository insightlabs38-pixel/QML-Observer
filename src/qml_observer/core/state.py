"""Internal rolling state for a `QMLMonitor` run.

Milestone 2, Issue #12 ("Implement rolling state").

`RunState` is *not* the statistics engine's `RollingWindow`
(`statistics/rolling.py`, Milestone 3, Issue #23) -- it lives one layer up.
It holds the bounded history of raw `StepObservation`s the monitor has
recorded for the current run, plus run-level bookkeeping (lifecycle flags,
start/end time, total step count). The statistics engine and the future
diagnosis engine will read from this window; this module has no detection
or statistical logic of its own.

Thread-safety (addendum, Concurrency / Distributed Training): `RunState`
and `QMLMonitor` are **not thread-safe** in v0.1. Concurrent `record()`
calls from multiple threads/processes are unsupported and may corrupt the
window. For multi-process/distributed training, use one monitor per
process/rank and aggregate reports post-hoc; true distributed-aware
monitoring is a post-1.0 feature.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from qml_observer.core.events import StepObservation
from qml_observer.schemas.diagnosis import DiagnosisResult


@dataclass
class RunState:
    """Bounded, mutable state for one monitor run.

    Not thread-safe -- see module docstring.

    Attributes:
        run_id: Identifier of the run this state belongs to.
        window_size: Maximum number of recent `StepObservation`s retained
            in the rolling window. Must be a positive int.
        planned_steps: Optional total steps the caller expects this run to
            take, used later by the compute-saved estimate (Milestone 7).
        started: Whether `QMLMonitor.start()` has been called for this run.
        finished: Whether `QMLMonitor.finish()` has been called for this run.
        start_time: Unix timestamp when the run started, if started.
        end_time: Unix timestamp when the run finished, if finished.
        latest_diagnosis: The most recent `DiagnosisResult` produced for
            this run, or None if no step has been diagnosed yet.
    """

    run_id: str
    window_size: int
    planned_steps: int | None = None

    started: bool = False
    finished: bool = False
    start_time: float | None = None
    end_time: float | None = None
    latest_diagnosis: DiagnosisResult | None = None

    _observations: deque = field(init=False, repr=False)
    _total_steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.window_size, int) or isinstance(self.window_size, bool):
            raise TypeError(f"window_size must be an int, got {type(self.window_size)!r}")
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        self._observations = deque(maxlen=self.window_size)

    def record(self, observation: StepObservation) -> None:
        """Append an observation to the rolling window and bump the step count."""
        if not isinstance(observation, StepObservation):
            raise TypeError(f"observation must be a StepObservation, got {type(observation)!r}")
        self._observations.append(observation)
        self._total_steps += 1

    @property
    def step_count(self) -> int:
        """Total number of steps recorded this run (not limited by window_size)."""
        return self._total_steps

    @property
    def window(self) -> list[StepObservation]:
        """The current bounded window of observations, oldest first."""
        return list(self._observations)

    @property
    def latest_observation(self) -> StepObservation | None:
        """The most recently recorded observation, or None if empty."""
        return self._observations[-1] if self._observations else None

    def mean_wall_time(self) -> float | None:
        """Mean `wall_time` across windowed observations that recorded one.

        Returns None if no observation in the window has a `wall_time`.
        Used by the future compute-saved estimate (addendum, resolved
        item): ``saved = (planned_total_steps - actual_steps) * mean_wall_time_per_step``.
        """
        wall_times = [
            obs.training_event.wall_time
            for obs in self._observations
            if obs.training_event.wall_time is not None
        ]
        if not wall_times:
            return None
        return sum(wall_times) / len(wall_times)

    def seed_step_count(self, step: int) -> None:
        """Advance the step counter to `step` without replaying any history.

        Milestone 13, Issue #97 ("Automatic resume"): used only when
        reconstructing a `QMLMonitor` from a `PausedRunSnapshot`
        (`recovery.resume.resume_monitor_from_snapshot`) so a resumed
        run's step numbering, `planned_steps`-based compute-saved
        estimate, and window-size behavior stay consistent with the
        paused run -- **not** a general-purpose way to rewind/fast-forward
        state. Only valid on a freshly-constructed or freshly-`reset()`
        `RunState` (no observations recorded yet); calling it after
        `record()` has already been used would silently misrepresent the
        window's actual history, so it raises instead.

        Args:
            step: The step count to seed. Must be >= 0.

        Raises:
            ValueError: If `step < 0`, or if this `RunState` already has
                recorded observations (`step_count != 0`).
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        if self._total_steps != 0:
            raise ValueError(
                "seed_step_count() can only be called on a RunState with no recorded "
                f"observations yet (step_count is currently {self._total_steps}, not 0)."
            )
        self._total_steps = step

    def reset(self) -> None:
        """Clear all recorded observations and lifecycle state in place.

        `run_id`, `window_size`, and `planned_steps` are left untouched --
        callers that want a fresh run_id should assign one before or after
        calling this (see `QMLMonitor.reset`).
        """
        self._observations = deque(maxlen=self.window_size)
        self._total_steps = 0
        self.started = False
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.latest_diagnosis = None
