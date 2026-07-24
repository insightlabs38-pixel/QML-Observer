"""LogAction: record a diagnosis, without otherwise intervening.

Milestone 5 (Volume XI), Issue #34 ("Implement logging action").

This is level 1 of the intervention model (plan.md §7): "Record
diagnosis, do not intervene." Every `ActionPolicy` mode (Issue #37) falls
back to `LogAction` for the quiet, common case (e.g. a healthy,
`"info"`-severity step) -- it is the one action that always executes,
never no-ops, and never raises.
"""

from __future__ import annotations

import logging

from qml_observer.actions.base import Action, ActionResult
from qml_observer.diagnosis.explanations import explain
from qml_observer.schemas.diagnosis import DiagnosisResult

_logger = logging.getLogger("qml_observer.actions")

#: Maps `DiagnosisResult.severity` to a `logging` level, so degraded or
#: critical diagnoses are visible in standard log aggregation without any
#: extra configuration.
_LEVEL_FOR_SEVERITY: dict[str, int] = {
    "info": logging.INFO,
    "warning": logging.WARNING,
    "critical": logging.ERROR,
}


class LogAction(Action):
    """Writes a one-line summary of `diagnosis` to the qml_observer logger.

    Never raises: a failure in the underlying `logging` call (e.g. a
    misconfigured handler) is caught and reported via `ActionResult`
    rather than propagated, per the fail-open action-safety contract
    (Issue #40).
    """

    name = "log"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Create a `LogAction`.

        Args:
            logger: Logger to write to. Defaults to the shared
                `"qml_observer.actions"` logger.
        """
        self._logger = logger if logger is not None else _logger

    def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
        """Log a one-line summary of `diagnosis` and return the outcome."""
        level = _LEVEL_FOR_SEVERITY.get(diagnosis.severity, logging.INFO)
        degraded_suffix = " [DEGRADED]" if diagnosis.degraded else ""
        summary = (
            f"issue={diagnosis.issue.value} confidence={diagnosis.confidence:.4f} "
            f"severity={diagnosis.severity}{degraded_suffix}"
        )
        try:
            self._logger.log(level, "qml_observer diagnosis: %s", summary)
        except Exception as exc:  # pragma: no cover - defensive, logging rarely fails
            return ActionResult(
                action_name=self.name,
                executed=False,
                message=f"LogAction failed to write to logger: {type(exc).__name__}: {exc}",
            )
        return ActionResult(
            action_name=self.name,
            executed=True,
            message=f"Logged diagnosis ({summary}).",
        )

    def render(self, diagnosis: DiagnosisResult) -> str:
        """Return the full human-readable explanation (not just the summary).

        Convenience for callers (CLI, reports) that want the richer
        `diagnosis.explanations.explain()` rendering rather than the
        compact one-liner written to the logger by `execute()`.
        """
        return explain(diagnosis)
