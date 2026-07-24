"""Action interface shared by all response actions.

Milestone 5 (Volume XI), Issue #33 ("Implement Action interface").

Every concrete action (`LogAction`, `AlertAction`, `StopAction`, and
future `PauseAction`/`RecoveryAction`) implements this same one-method
contract so `ActionPolicy` (Issue #37) can select and run an arbitrary
action without special-casing, exactly as `BaseDetector` lets
`DiagnosisEngine` drive an arbitrary list of detectors uniformly
(detectors/base.py).

Core principle (plan.md §2, "non-invasive monitoring layer"): an action
never reaches into the caller's training loop and forces it to stop --
it can only *signal* what the policy recommends (log a line, print a
terminal alert, flip a flag the caller is expected to check via
`QMLMonitor.should_stop()`). This mirrors the fail-open policy elsewhere
in the project (addendum §1): qml_observer observes and recommends, the
user's code remains in control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from qml_observer.schemas._validation import check_non_empty_str, check_type
from qml_observer.schemas.diagnosis import DiagnosisResult


@dataclass
class ActionResult:
    """Outcome of executing a single `Action`.

    Attributes:
        action_name: Stable identifier of the action that produced this
            result (e.g. `"log"`, `"alert"`, `"stop"`), matching that
            action's `name` attribute.
        executed: Whether the action actually performed its intended side
            effect. Some actions deliberately no-op for certain
            diagnoses (e.g. `AlertAction` skips alerting on a healthy,
            `"info"`-severity result) -- `executed=False` distinguishes a
            deliberate no-op from a real action, for callers/tests that
            care (e.g. Issue #40, action-safety tests).
        message: Human-readable description of what happened (or why the
            action was skipped), suitable for logs or CLI echo.
    """

    action_name: str
    executed: bool
    message: str

    def __post_init__(self) -> None:
        check_non_empty_str(self.action_name, "action_name")
        check_type(self.executed, bool, "executed")
        check_type(self.message, str, "message")


class Action(ABC):
    """Abstract interface every concrete action must implement.

    Actions are stateless from the `execute()` caller's point of view
    (each call is handed the full `DiagnosisResult` it needs), though a
    concrete action may keep its own internal bookkeeping (e.g.
    `StopAction` remembers whether a stop was ever requested, so callers
    can check it after the fact).
    """

    #: Stable identifier for this action, used in `ActionResult.action_name`
    #: and by `ActionPolicy`/reporting to refer to this action by name.
    #: Concrete subclasses must override this.
    name: str = "base"

    @abstractmethod
    def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
        """Carry out this action's response to `diagnosis`.

        Must never raise for a well-formed `DiagnosisResult`: an action
        that cannot safely perform its side effect (e.g. a broken log
        handler) should catch its own internal errors and return
        `executed=False` with an explanatory `message`, consistent with
        the project's fail-open philosophy -- an action's own failure
        must never propagate into the caller's training loop any more
        than a detector's would (Issue #40, action-safety tests).
        """
        raise NotImplementedError
