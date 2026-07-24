"""ActionPolicy: select which Action to run for a given diagnosis.

Milestone 5 (Volume XI), Issue #37 ("Implement ActionPolicy"), Issue #38
("Add warn mode"), Issue #39 ("Add stop mode").

`ActionPolicy` is the seam between the diagnosis layer (Milestone 4) and
the action layer (`actions/log.py`, `actions/alert.py`, `actions/stop.py`):
given a `DiagnosisResult`, it decides *which* `Action` should run, while
each `Action` itself only knows how to run once selected. This mirrors
the diagnosis-engine/detector split (blueprint's second architectural
rule): a policy decides "what to do about it", never how to log/alert/
stop.

Scope note: this module supports the `"log"`, `"warn"`, and `"stop"`
modes (Issues #37-#39). `"pause"` and `"adaptive"` are accepted as valid
mode strings -- matching `QMLMonitor`'s `_VALID_POLICIES` so a
`QMLMonitor(policy=...)` value is always constructible as an
`ActionPolicy(mode=...)` too -- but until `PauseAction` ships (blueprint
Volume XIV, Milestone 13, no issue number yet assigned), `"pause"`
behaves identically to `"warn"` (log + alert, never an automatic stop):
this is the conservative choice plan.md §7 calls for ("The default
should be conservative"), not a placeholder that silently does nothing.

Degraded-diagnosis safety (addendum §1): a `degraded=True` diagnosis
never selects `StopAction`, regardless of `mode`, *unless* the caller
constructed this policy with `mode="adaptive"` **and**
`allow_stop_on_degraded=True` -- the explicit two-part opt-in the
addendum describes as "mode='adaptive' with a flag acknowledging this
risk". Without that flag, `mode="adaptive"` behaves exactly like
`"stop"`.
"""

from __future__ import annotations

from qml_observer.actions.alert import AlertAction
from qml_observer.actions.base import Action, ActionResult
from qml_observer.actions.log import LogAction
from qml_observer.actions.stop import StopAction
from qml_observer.schemas.diagnosis import DiagnosisResult

#: Supported modes. Kept identical to `qml_observer.core.monitor._VALID_POLICIES`
#: so every valid `QMLMonitor(policy=...)` value is a valid `ActionPolicy(mode=...)`
#: too, even though `"pause"`/`"adaptive"`'s full behavior ships incrementally
#: (see module docstring).
VALID_MODES = frozenset({"log", "warn", "pause", "stop", "adaptive"})

#: Modes that may ever select `StopAction` for a non-degraded, critical diagnosis.
_STOP_CAPABLE_MODES = frozenset({"stop", "adaptive"})


class ActionPolicy:
    """Chooses and (optionally) runs an `Action` for each `DiagnosisResult`.

    Example:
        >>> policy = ActionPolicy(mode="stop")
        >>> result = policy.execute(diagnosis)
        >>> if policy.stop_action.triggered:
        ...     break  # in the caller's own training loop
    """

    def __init__(
        self,
        mode: str = "warn",
        *,
        allow_stop_on_degraded: bool = False,
        log_action: LogAction | None = None,
        alert_action: AlertAction | None = None,
        stop_action: StopAction | None = None,
    ) -> None:
        """Create an `ActionPolicy`.

        Args:
            mode: One of `"log"`, `"warn"`, `"pause"`, `"stop"`,
                `"adaptive"` (see module docstring for `"pause"`/
                `"adaptive"` scope notes).
            allow_stop_on_degraded: When `True` **and** `mode="adaptive"`,
                allows a `degraded=True` diagnosis to select `StopAction`
                for a critical severity, overriding the default
                conservative behavior (addendum §1). Ignored for every
                other mode: degraded diagnoses never select `StopAction`
                outside this explicit combination.
            log_action: `LogAction` instance to use. Defaults to a new
                `LogAction()`.
            alert_action: `AlertAction` instance to use. Defaults to a
                new `AlertAction()`.
            stop_action: `StopAction` instance to use. Defaults to a new
                `StopAction()`.

        Raises:
            ValueError: If `mode` is not one of `VALID_MODES`.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

        self._mode = mode
        self._allow_stop_on_degraded = allow_stop_on_degraded
        self._log_action = log_action if log_action is not None else LogAction()
        self._alert_action = alert_action if alert_action is not None else AlertAction()
        self._stop_action = stop_action if stop_action is not None else StopAction()

    @property
    def mode(self) -> str:
        """The configured mode."""
        return self._mode

    @property
    def allow_stop_on_degraded(self) -> bool:
        """Whether `mode="adaptive"` is permitted to stop on a degraded result."""
        return self._allow_stop_on_degraded

    @property
    def stop_action(self) -> StopAction:
        """The `StopAction` instance this policy uses, for `.triggered` checks."""
        return self._stop_action

    def select_action(self, diagnosis: DiagnosisResult) -> Action:
        """Return the `Action` this policy recommends for `diagnosis`.

        Does not execute the action; see `execute()` for that. Exposed
        separately so callers/tests can inspect the decision (e.g. which
        action *would* run) without triggering its side effects.
        """
        if self._mode == "log":
            return self._log_action

        opted_into_degraded_stop = self._mode == "adaptive" and self._allow_stop_on_degraded
        can_stop = (
            self._mode in _STOP_CAPABLE_MODES
            and diagnosis.severity == "critical"
            and (not diagnosis.degraded or opted_into_degraded_stop)
        )
        if can_stop:
            return self._stop_action

        # warn / pause / stop / adaptive all fall back to alerting for
        # anything non-critical (or critical-but-degraded-and-not-opted-in),
        # and to a quiet log for genuinely uninteresting ("info") results.
        if diagnosis.severity == "info":
            return self._log_action
        return self._alert_action

    def execute(self, diagnosis: DiagnosisResult) -> ActionResult:
        """Select and run the recommended action for `diagnosis`."""
        return self.select_action(diagnosis).execute(diagnosis)

    def reset(self) -> None:
        """Clear any stateful action memory (currently: `StopAction.triggered`)."""
        self._stop_action.reset()
