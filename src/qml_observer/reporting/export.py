"""Compute-saved estimation and simple summary export.

Milestone 7 (Volume XII/XIX-XX), Issue #51 ("Compute-saved estimation").

Formula (addendum §11, "Resolved Items" -- authoritative):

    saved = (planned_total_steps - actual_steps_at_stop) * mean_wall_time_per_step

`planned_total_steps` is either user-supplied (`QMLMonitor(planned_steps=...)`,
already threaded through `core/monitor.py` and `core/state.py` since
Milestone 2) or unavailable, in which case no estimate can be produced
(`None`, not a guessed default -- silently fabricating a number here would
undermine the project's "scientifically falsifiable" positioning, blueprint
Volume XX/addendum §3). `mean_wall_time_per_step` comes from
`RunState.mean_wall_time()`, already implemented for exactly this purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qml_observer.core.state import RunState


def estimate_compute_saved(
    planned_steps: int | None,
    actual_steps: int,
    mean_wall_time_per_step: float | None,
) -> float | None:
    """Estimate wall-clock compute saved by stopping early, in seconds.

    Args:
        planned_steps: The total steps the run was expected to take, or
            `None` if unknown/not configured.
        actual_steps: The number of steps actually recorded before
            stopping (e.g. `state.step_count`, or an explicit earlier
            step index if the caller wants the estimate as of an earlier
            point in the run).
        mean_wall_time_per_step: Mean wall-clock time per step observed so
            far (e.g. `RunState.mean_wall_time()`), or `None` if no
            per-step timing has been recorded.

    Returns:
        `(planned_steps - actual_steps) * mean_wall_time_per_step`,
        clamped to `0.0` if the run already reached or exceeded
        `planned_steps` (stopping "early" saved nothing). `None` if either
        `planned_steps` or `mean_wall_time_per_step` is unavailable --
        never a fabricated guess.

    Raises:
        ValueError: If `actual_steps` is negative.
    """
    if actual_steps < 0:
        raise ValueError(f"actual_steps must be >= 0, got {actual_steps}")
    if planned_steps is None or mean_wall_time_per_step is None:
        return None
    remaining = planned_steps - actual_steps
    if remaining <= 0:
        return 0.0
    return remaining * mean_wall_time_per_step


def estimate_compute_saved_from_state(
    state: RunState,
    *,
    actual_steps_at_stop: int | None = None,
) -> float | None:
    """Convenience wrapper around `estimate_compute_saved` reading directly
    from a `RunState` (e.g. `monitor.state`).

    Args:
        state: The `RunState` to read `planned_steps` and
            `mean_wall_time()` from.
        actual_steps_at_stop: Override for the "actual steps" term, e.g.
            to compute the estimate as of an earlier step than
            `state.step_count` currently reflects. Defaults to
            `state.step_count`.
    """
    actual = actual_steps_at_stop if actual_steps_at_stop is not None else state.step_count
    return estimate_compute_saved(state.planned_steps, actual, state.mean_wall_time())


def format_compute_saved(seconds: float | None) -> str:
    """Render a compute-saved estimate as a short human-readable string.

    Matches the blueprint's Volume XV/XX CLI-output style (e.g.
    `"~2.4 hours"`, `"~3h 32m"`-equivalent granularity, simplified to a
    single unit). Returns `"unknown (no planned_steps configured)"` when
    `seconds` is `None`, so CLI/report output never silently shows a blank
    or a misleading `"0"`.
    """
    if seconds is None:
        return "unknown (no planned_steps configured)"
    if seconds < 1:
        return "~0 seconds"
    minutes = seconds / 60
    if minutes < 1:
        return f"~{seconds:.0f} seconds"
    hours = minutes / 60
    if hours < 1:
        return f"~{minutes:.1f} minutes"
    return f"~{hours:.1f} hours"


def export_summary_json(summary: dict[str, Any], path: str | Path) -> Path:
    """Write a run-summary dict (see `reporting.summary.build_run_summary`)
    to `path` as a single formatted JSON document. Returns the resolved
    `Path` for convenience."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved
