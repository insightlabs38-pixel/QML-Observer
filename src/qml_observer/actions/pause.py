"""PauseAction: signal that training should be paused, preserving resumable state.

Milestone 13 (blueprint Volume XIV), Issue #90b ("Implement PauseAction
itself"). Added *before* Issues #90-#97 per `future_milestones_plan.md`'s
resequencing note: those issues cover *recovery strategies*, none of which
ship a real pause-and-preserve-state action of their own, and Issue #97
("automatic resume") is meaningless without something to resume *from*.

This is level 3 of the intervention model (plan.md §7): "Stop the
optimization loop but preserve state." Until this issue, `"pause"` mode
behaved identically to `"warn"` (`docs/architecture/actions.md`) -- a
deliberate conservative placeholder, not a silent no-op. `PauseAction` now
gives `"pause"` real, distinct behavior: like `StopAction`, it can only
*signal* a pause request via a `.triggered` flag for the caller's own loop
to check (the non-invasive core principle, plan.md §2, `actions/base.py`);
unlike `StopAction`, it also captures a `PausedRunSnapshot` -- enough of
the run's configuration and rolling state to reconstruct a `QMLMonitor`
later (Issue #97, `RecoveryExecutor.apply`/automatic resume) without
requiring the caller to keep anything around themselves.

Degraded-diagnosis safety mirrors `StopAction`/`ActionPolicy` (addendum
§1): `ActionPolicy` never selects `PauseAction` for a `degraded=True`
diagnosis except under the same `mode="adaptive"` +
`allow_stop_on_degraded=True` opt-in already used for `StopAction` -- see
`actions/policies.py`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from qml_observer.actions.base import Action, ActionResult
from qml_observer.diagnosis.explanations import explain
from qml_observer.schemas.diagnosis import DiagnosisResult

_logger = logging.getLogger("qml_observer.actions")


@dataclass
class PausedRunSnapshot:
    """Enough state to describe (and later resume) a paused run.

    Deliberately a plain, JSON-serializable-friendly dataclass (no live
    object references such as a `RunState` deque or a `QMLMonitor`
    instance) so it can be logged, written to a report, or handed to a
    future `RecoveryExecutor` (Issue #97) without pinning down how the
    caller's own process/training loop is structured.

    Attributes:
        run_id: Identifier of the paused run.
        step: The step index (`RunState.step_count`) at the moment of pause.
        paused_at: Unix timestamp when the pause was requested.
        diagnosis: The `DiagnosisResult` that triggered the pause.
        window_size: The monitor's configured rolling-window size, needed
            to reconstruct an equivalent `QMLMonitor` on resume.
        planned_steps: The monitor's configured `planned_steps`, if any,
            carried over so a resumed run's compute-saved estimate stays
            consistent.
        extra: Optional caller-supplied context (e.g. optimizer state
            checkpoint path, random seed) that qml_observer does not
            interpret itself but preserves for the resume path.
    """

    run_id: str
    step: int
    paused_at: float
    diagnosis: DiagnosisResult
    window_size: int
    planned_steps: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class PauseAction(Action):
    """Records a pause request and captures a resumable `PausedRunSnapshot`.

    Stateful across calls, like `StopAction`: once `execute()` has been
    called, `triggered` (and `paused`) stay `True` until `resume()` or
    `reset()` is called, so a caller polling `.triggered` between steps
    reliably observes a pause request.

    Unlike `StopAction`, `PauseAction` is explicitly designed to be
    reversible: `resume()` clears the pause flag (but keeps the last
    snapshot available via `last_snapshot` for inspection/logging) so the
    same monitor/action instance can be paused and resumed repeatedly
    within one run, rather than requiring a full `reset()`.
    """

    name = "pause"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger if logger is not None else _logger
        self._triggered = False
        self._last_diagnosis: DiagnosisResult | None = None
        self._last_snapshot: PausedRunSnapshot | None = None

    @property
    def triggered(self) -> bool:
        """Whether a pause has been requested since construction/`resume()`/`reset()`."""
        return self._triggered

    @property
    def paused(self) -> bool:
        """Alias for `.triggered`, matching the "paused" vocabulary elsewhere."""
        return self._triggered

    @property
    def last_diagnosis(self) -> DiagnosisResult | None:
        """The `DiagnosisResult` that most recently requested a pause, if any."""
        return self._last_diagnosis

    @property
    def last_snapshot(self) -> PausedRunSnapshot | None:
        """The `PausedRunSnapshot` captured by the most recent pause request.

        Remains available after `resume()` (until the *next* pause or a
        `reset()`) so callers can still inspect what was paused even after
        resuming, e.g. for a recovery/audit log.
        """
        return self._last_snapshot

    def execute(
        self,
        diagnosis: DiagnosisResult,
        *,
        run_id: str | None = None,
        step: int | None = None,
        window_size: int | None = None,
        planned_steps: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ActionResult:
        """Record a pause request for `diagnosis` and capture a snapshot.

        The keyword-only run-context arguments are optional so
        `PauseAction` still satisfies the plain `Action.execute(diagnosis)`
        contract used by `ActionPolicy` (which does not currently pass
        them); `QMLMonitor` passes them explicitly so the captured
        snapshot is actually useful for a future resume, rather than an
        empty placeholder.

        Never raises: a failure while logging is caught and reported via
        `ActionResult` rather than propagated (Issue #40, action safety),
        and the pause is still recorded regardless -- a logging failure
        must never suppress a real pause request.
        """
        self._triggered = True
        self._last_diagnosis = diagnosis
        self._last_snapshot = PausedRunSnapshot(
            run_id=run_id if run_id is not None else "unknown",
            step=step if step is not None else 0,
            paused_at=time.time(),
            diagnosis=diagnosis,
            window_size=window_size if window_size is not None else 0,
            planned_steps=planned_steps,
            extra=dict(extra) if extra else {},
        )

        message = (
            f"Pause requested for issue={diagnosis.issue.value} "
            f"confidence={diagnosis.confidence:.4f}. State captured in "
            "`.last_snapshot` for a future resume; the caller is responsible for "
            "checking `.triggered` (or QMLMonitor.should_pause()) and "
            "pausing its own training loop -- qml_observer cannot do this on the "
            "caller's behalf."
        )
        try:
            self._logger.warning("qml_observer PAUSE requested:\n%s", explain(diagnosis))
        except Exception as exc:  # pragma: no cover - defensive; pause is still recorded above
            return ActionResult(
                action_name=self.name,
                executed=True,
                message=f"{message} (Note: logging the pause failed: {type(exc).__name__}: {exc})",
            )

        return ActionResult(action_name=self.name, executed=True, message=message)

    def resume(self) -> None:
        """Clear the pause request while keeping `last_snapshot`/`last_diagnosis`.

        Use this (rather than `reset()`) when the caller's training loop is
        actually resuming, so history of the pause remains inspectable.
        """
        self._triggered = False

    def reset(self) -> None:
        """Clear the pause request and all captured history.

        Used when a `QMLMonitor` is `reset()` for an entirely new run, as
        opposed to `resume()`-ing the same one.
        """
        self._triggered = False
        self._last_diagnosis = None
        self._last_snapshot = None
