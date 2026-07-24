"""StopAction: signal that training should be terminated.

Milestone 5 (Volume XI), Issue #36 ("Implement stop action").

This is level 4 of the intervention model (plan.md §7): "Terminate the
run and persist diagnostics." Per the non-invasive core principle
(plan.md §2) and `actions/base.py`'s module docstring, `StopAction`
cannot itself reach into the caller's training loop -- it can only
record that a stop was requested and expose that fact for the caller (or
`QMLMonitor.should_stop()`, wired up in a later issue) to check and act
on, typically by `break`-ing out of its own loop.
"""

from __future__ import annotations

import logging

from qml_observer.actions.base import Action, ActionResult
from qml_observer.diagnosis.explanations import explain
from qml_observer.schemas.diagnosis import DiagnosisResult

_logger = logging.getLogger("qml_observer.actions")


class StopAction(Action):
    """Records a stop request and exposes it via `triggered`.

    Stateful across calls (unlike `LogAction`/`AlertAction`): once
    `execute()` has been called, `triggered` stays `True` until `reset()`
    is called, so a caller polling `.triggered` between steps reliably
    observes a stop request even if it checks less often than
    `execute()` is called.
    """

    name = "stop"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger if logger is not None else _logger
        self._triggered = False
        self._last_diagnosis: DiagnosisResult | None = None

    @property
    def triggered(self) -> bool:
        """Whether a stop has been requested since construction/`reset()`."""
        return self._triggered

    @property
    def last_diagnosis(self) -> DiagnosisResult | None:
        """The `DiagnosisResult` that most recently requested a stop, if any."""
        return self._last_diagnosis

    def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
        """Record a stop request for `diagnosis`.

        Never raises: a failure while logging is caught and reported via
        `ActionResult` rather than propagated (Issue #40), and the stop
        is still recorded regardless -- a logging failure must never
        suppress a real stop request.
        """
        self._triggered = True
        self._last_diagnosis = diagnosis

        message = (
            f"Stop requested for issue={diagnosis.issue.value} "
            f"confidence={diagnosis.confidence:.4f}. The caller is responsible for "
            "checking `.triggered` (or QMLMonitor.should_stop()) and breaking its "
            "training loop; qml_observer cannot do this on the caller's behalf."
        )
        try:
            self._logger.error("qml_observer STOP requested:\n%s", explain(diagnosis))
        except Exception as exc:  # pragma: no cover - defensive; stop is still recorded above
            return ActionResult(
                action_name=self.name,
                executed=True,
                message=f"{message} (Note: logging the stop failed: {type(exc).__name__}: {exc})",
            )

        return ActionResult(action_name=self.name, executed=True, message=message)

    def reset(self) -> None:
        """Clear the stop request, e.g. when a `QMLMonitor` is `reset()`."""
        self._triggered = False
        self._last_diagnosis = None
