"""AlertAction: emit a visible warning for a non-healthy diagnosis.

Milestone 5 (Volume XI), Issue #35 ("Implement warning action").

This is level 2 of the intervention model (plan.md §7): "Emit terminal
warning, dashboard alert, or webhook notification." The MVP (plan.md §11)
scopes this to terminal alerts only; dashboard/webhook delivery is
Milestone 10/11 (`integrations/webhook.py`), which will reuse this same
`Action` (or a sibling built on top of it) rather than replace it.

`AlertAction` deliberately no-ops (returns `executed=False`) for
`severity == "info"` results (i.e. `HEALTHY`, `CONVERGED`, or
`INSUFFICIENT_EVIDENCE` with nothing to report) -- alerting on every
healthy step would train users to ignore the alert stream entirely,
which defeats its purpose.
"""

from __future__ import annotations

import logging
import sys
from typing import Protocol

from qml_observer.actions.base import Action, ActionResult
from qml_observer.diagnosis.explanations import explain
from qml_observer.schemas.diagnosis import DiagnosisResult

_logger = logging.getLogger("qml_observer.actions")

_BANNER = "=" * 60


class _WritableStream(Protocol):
    """Minimal structural type for `AlertAction`'s `stream` parameter.

    Anything with a `.write(str)` method qualifies (e.g. `sys.stderr`,
    `io.StringIO`, or a test double) -- `AlertAction` never relies on any
    other file-object behavior.
    """

    def write(self, text: str) -> object: ...


class AlertAction(Action):
    """Prints a terminal alert and logs at `WARNING`/`ERROR` for `diagnosis`.

    Never raises: any failure while writing to `stream` or `logger` is
    caught and reported via `ActionResult` (Issue #40, action safety).
    """

    name = "alert"

    def __init__(
        self,
        stream: _WritableStream | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Create an `AlertAction`.

        Args:
            stream: A file-like object with a `.write()` method to print
                the terminal alert to. Defaults to `sys.stderr`, since
                alerts are operational signal, not program output.
            logger: Logger to additionally record the alert to. Defaults
                to the shared `"qml_observer.actions"` logger.
        """
        self._stream = stream if stream is not None else sys.stderr
        self._logger = logger if logger is not None else _logger

    def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
        """Emit a terminal alert for `diagnosis`, unless it is merely `"info"`."""
        if diagnosis.severity == "info":
            return ActionResult(
                action_name=self.name,
                executed=False,
                message="Skipped: diagnosis severity is 'info' (nothing to alert on).",
            )

        body = explain(diagnosis)
        alert_text = f"\n{_BANNER}\nQML OBSERVER ALERT\n{_BANNER}\n{body}\n{_BANNER}"
        try:
            print(alert_text, file=self._stream)
        except Exception as exc:
            return ActionResult(
                action_name=self.name,
                executed=False,
                message=f"AlertAction failed to write to stream: {type(exc).__name__}: {exc}",
            )

        level = logging.ERROR if diagnosis.severity == "critical" else logging.WARNING
        try:
            self._logger.log(
                level,
                "qml_observer ALERT: issue=%s confidence=%.4f severity=%s",
                diagnosis.issue.value,
                diagnosis.confidence,
                diagnosis.severity,
            )
        except Exception:  # pragma: no cover - defensive; the terminal alert already fired
            pass

        return ActionResult(
            action_name=self.name,
            executed=True,
            message=f"Alerted on {diagnosis.issue.value} (severity={diagnosis.severity}).",
        )
