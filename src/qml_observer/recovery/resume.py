"""Automatic resume: reconstruct a `QMLMonitor` from a `PausedRunSnapshot`.

Milestone 13, Issue #97 ("Automatic resume").

Scope note, stated plainly: qml_observer does not own the caller's
training loop (plan.md §2, non-invasive core principle), so "automatic
resume" here means what it can honestly mean for a *non-invasive
observability layer* -- automatically reconstructing an equivalent
`QMLMonitor` (same `run_id`, `window_size`, `planned_steps`, and step
counter) from a `PausedRunSnapshot`, so the caller's own resumed training
loop can immediately resume being monitored at the correct step number,
without hand-reconstructing that configuration themselves. It does
**not** mean qml_observer restarts the caller's quantum computation,
optimizer, or parameter values -- those remain entirely the caller's
responsibility, exactly as `PauseAction`'s own docstring already states
for pausing itself.

This is deliberately the last piece of the `PauseAction` -> `RecoveryPlanner`
/`RecoveryExecutor` -> `RecoveryEvaluator` -> resume chain: Issue #90b
captured a resumable snapshot, #90/#91-#96b propose and apply changes,
#96 judges whether they helped, and this module closes the loop by making
"pick back up monitoring where we left off" a single function call instead
of the caller re-deriving `window_size`/`planned_steps`/step-numbering by
hand from the snapshot.
"""

from __future__ import annotations

from typing import Any

from qml_observer.actions.pause import PausedRunSnapshot
from qml_observer.actions.policies import ActionPolicy
from qml_observer.core.monitor import QMLMonitor
from qml_observer.detectors.base import BaseDetector


def resume_monitor_from_snapshot(
    snapshot: PausedRunSnapshot,
    *,
    detectors: list[BaseDetector] | None = None,
    policy: str = "pause",
    action_policy: ActionPolicy | None = None,
    reporter: Any | None = None,
) -> QMLMonitor:
    """Reconstruct a `QMLMonitor` continuing from `snapshot`.

    The returned monitor has the same `run_id`, `window_size`, and
    `planned_steps` as the paused run, and its step counter is seeded to
    `snapshot.step` so the next `update(step=snapshot.step, ...)` call
    continues the step sequence correctly (and so `planned_steps`-based
    compute-saved estimates stay meaningful). The caller then drives it
    exactly like any other `QMLMonitor` -- calling `update()` per step (it
    auto-starts on first `update()`, same as a brand-new monitor).

    Args:
        snapshot: The `PausedRunSnapshot` captured by a `PauseAction`
            (`monitor.action_policy.pause_action.last_snapshot`).
        detectors: `BaseDetector` instances for the resumed monitor.
            Not part of `PausedRunSnapshot` (detectors are stateful,
            per-run objects, not serializable run *configuration*), so
            the caller must re-supply them -- typically the same
            detector instances/configuration used before the pause.
        policy: Action policy mode for the resumed monitor. Defaults to
            `"pause"` so a resumed run can be paused again if the same
            issue recurs. Ignored if `action_policy` is given.
        action_policy: A pre-configured `ActionPolicy` to use instead of
            building one from `policy`, same as `QMLMonitor.__init__`.
        reporter: Optional reporter for the resumed monitor, same as
            `QMLMonitor.__init__`.

    Returns:
        A new `QMLMonitor` ready to continue the paused run.

    Raises:
        TypeError: If `snapshot` is not a `PausedRunSnapshot`.

    Known limitations (see `docs/architecture/recovery.md`):
        - The rolling window of prior `StepObservation`s (recent
          gradients/loss values) is **not** restored -- `PausedRunSnapshot`
          deliberately does not capture it (it is not a lightweight,
          JSON-serializable-friendly field, per `PausedRunSnapshot`'s own
          docstring). The resumed monitor's window starts empty and
          repopulates as new steps are recorded; any detector relying on
          `patience`/persistence across the pause boundary will need that
          many new steps again before it can trigger.
        - Wall-clock duration is not preserved across the pause: the
          resumed monitor's `start_time` is set on its next `update()`
          call (auto-start), not backdated to the original run's start.
          Any compute-saved estimate computed after resuming reflects
          only time elapsed *after* the resume, not the full run.
    """
    if not isinstance(snapshot, PausedRunSnapshot):
        raise TypeError(f"snapshot must be a PausedRunSnapshot, got {type(snapshot)!r}")

    monitor = QMLMonitor(
        detectors=detectors,
        policy=policy,
        window_size=snapshot.window_size if snapshot.window_size > 0 else 100,
        run_id=snapshot.run_id if snapshot.run_id != "unknown" else None,
        reporter=reporter,
        planned_steps=snapshot.planned_steps,
        action_policy=action_policy,
    )
    monitor.state.seed_step_count(snapshot.step)
    return monitor
