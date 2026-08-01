"""ActionPolicy: select which Action to run for a given diagnosis.

Milestone 5 (Volume XI), Issue #37 ("Implement ActionPolicy"), Issue #38
("Add warn mode"), Issue #39 ("Add stop mode"). Milestone 13, Issue #90b
("Implement PauseAction itself") gives `"pause"` its own real behavior.

`ActionPolicy` is the seam between the diagnosis layer (Milestone 4) and
the action layer (`actions/log.py`, `actions/alert.py`, `actions/stop.py`,
`actions/pause.py`): given a `DiagnosisResult`, it decides *which* `Action`
should run, while each `Action` itself only knows how to run once
selected. This mirrors the diagnosis-engine/detector split (blueprint's
second architectural rule): a policy decides "what to do about it", never
how to log/alert/pause/stop.

Scope note: `"pause"` now selects a real `PauseAction` for a critical,
non-degraded diagnosis (Issue #90b) -- it is level 3 of the intervention
model (plan.md §7, "Stop the optimization loop but preserve state"),
between `"warn"` (level 2) and `"stop"` (level 4). It no longer falls
back to `AlertAction`'s behavior for critical diagnoses, though it still
does for anything less than critical, exactly like `"stop"` does.

Degraded-diagnosis safety (addendum §1): a `degraded=True` diagnosis
never selects `StopAction` *or* `PauseAction`, regardless of `mode`,
*unless* the caller constructed this policy with `mode="adaptive"`
**and** `allow_stop_on_degraded=True` -- the explicit two-part opt-in the
addendum describes as "mode='adaptive' with a flag acknowledging this
risk". Without that flag, `mode="adaptive"` behaves exactly like
`"stop"`.
"""

from __future__ import annotations

from qml_observer.actions.alert import AlertAction
from qml_observer.actions.base import Action, ActionResult
from qml_observer.actions.log import LogAction
from qml_observer.actions.pause import PauseAction
from qml_observer.actions.stop import StopAction
from qml_observer.schemas.diagnosis import DiagnosisResult

#: Supported modes. Kept identical to `qml_observer.core.monitor._VALID_POLICIES`
#: so every valid `QMLMonitor(policy=...)` value is a valid `ActionPolicy(mode=...)`
#: too.
VALID_MODES = frozenset({"log", "warn", "pause", "stop", "adaptive"})

#: Modes that may ever select `StopAction` for a non-degraded, critical diagnosis.
_STOP_CAPABLE_MODES = frozenset({"stop", "adaptive"})

#: Modes that may ever select `PauseAction` for a non-degraded, critical diagnosis.
#: `"stop"`/`"adaptive"` intentionally do NOT also pause -- a policy that can
#: already stop has no need for the intermediate pause level.
_PAUSE_CAPABLE_MODES = frozenset({"pause"})


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
        pause_action: PauseAction | None = None,
    ) -> None:
        """Create an `ActionPolicy`.

        Args:
            mode: One of `"log"`, `"warn"`, `"pause"`, `"stop"`,
                `"adaptive"` (see module docstring for `"pause"`/
                `"adaptive"` scope notes).
            allow_stop_on_degraded: When `True` **and** `mode="adaptive"`,
                allows a `degraded=True` diagnosis to select `StopAction`
                for a critical severity, overriding the default
                conservative behavior (addendum §1). Also applies to
                `PauseAction` under `mode="pause"` there is no equivalent
                escalation path, so this flag has no effect for that mode.
                Ignored for every other mode: degraded diagnoses never
                select `StopAction`/`PauseAction` outside this explicit
                combination.
            log_action: `LogAction` instance to use. Defaults to a new
                `LogAction()`.
            alert_action: `AlertAction` instance to use. Defaults to a
                new `AlertAction()`.
            stop_action: `StopAction` instance to use. Defaults to a new
                `StopAction()`.
            pause_action: `PauseAction` instance to use. Defaults to a
                new `PauseAction()`.

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
        self._pause_action = pause_action if pause_action is not None else PauseAction()

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

    @property
    def pause_action(self) -> PauseAction:
        """The `PauseAction` instance this policy uses, for `.triggered`/`.last_snapshot` checks."""
        return self._pause_action

    def select_action(self, diagnosis: DiagnosisResult) -> Action:
        """Return the `Action` this policy recommends for `diagnosis`.

        Does not execute the action; see `execute()` for that. Exposed
        separately so callers/tests can inspect the decision (e.g. which
        action *would* run) without triggering its side effects.
        """
        if self._mode == "log":
            return self._log_action

        opted_into_degraded_escalation = self._mode == "adaptive" and self._allow_stop_on_degraded
        can_stop = (
            self._mode in _STOP_CAPABLE_MODES
            and diagnosis.severity == "critical"
            and (not diagnosis.degraded or opted_into_degraded_escalation)
        )
        if can_stop:
            return self._stop_action

        can_pause = (
            self._mode in _PAUSE_CAPABLE_MODES
            and diagnosis.severity == "critical"
            and (not diagnosis.degraded or opted_into_degraded_escalation)
        )
        if can_pause:
            return self._pause_action

        # warn / pause / stop / adaptive all fall back to alerting for
        # anything non-critical (or critical-but-degraded-and-not-opted-in),
        # and to a quiet log for genuinely uninteresting ("info") results.
        if diagnosis.severity == "info":
            return self._log_action
        return self._alert_action

    def execute(
        self,
        diagnosis: DiagnosisResult,
        *,
        run_id: str | None = None,
        step: int | None = None,
        window_size: int | None = None,
        planned_steps: int | None = None,
    ) -> ActionResult:
        """Select and run the recommended action for `diagnosis`.

        The keyword-only run-context arguments are forwarded to
        `PauseAction.execute()` when it is selected (so its captured
        `PausedRunSnapshot` is actually populated); every other `Action`
        ignores them, since only `PauseAction.execute()` accepts them.
        """
        action = self.select_action(diagnosis)
        if action is self._pause_action:
            return self._pause_action.execute(
                diagnosis,
                run_id=run_id,
                step=step,
                window_size=window_size,
                planned_steps=planned_steps,
            )
        return action.execute(diagnosis)

    def reset(self) -> None:
        """Clear any stateful action memory (`StopAction`/`PauseAction` triggers)."""
        self._stop_action.reset()
        self._pause_action.reset()
