"""WandbTracker: write QMLMonitor output into an existing Weights & Biases run.

Milestone 14 (`future_milestones_plan.md`), Issue #101 ("Experiment-tracker
integrations").

Requires the optional `wandb` dependency (`pip install
qml-observer[wandb]`). Does *not* call `wandb.init()` itself -- attach to
a run you've already initialized yourself, same rationale as
`MLflowTracker`.
"""

from __future__ import annotations

from typing import Any

from qml_observer.integrations.trackers.base import BaseExperimentTracker

try:
    import wandb
except ImportError as _exc:  # pragma: no cover - exercised only without wandb installed
    wandb = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = _exc
else:
    _IMPORT_ERROR = None


def _require_wandb() -> None:
    if wandb is None:
        raise ImportError(
            "WandbTracker requires the optional 'wandb' dependency. "
            "Install it with `pip install qml-observer[wandb]` or "
            "`pip install wandb>=0.16`."
        ) from _IMPORT_ERROR


class WandbTracker(BaseExperimentTracker):
    """Forwards `QMLMonitor` events/diagnoses into a Weights & Biases run.

    Example:
        >>> import wandb
        >>> from qml_observer import QMLMonitor
        >>> from qml_observer.integrations.trackers.wandb_tracker import WandbTracker
        >>>
        >>> run = wandb.init(project="qml-experiments")
        >>> monitor = QMLMonitor(reporter=WandbTracker(run=run))
        >>> for step in range(1000):
        ...     monitor.update(step=step, loss=loss)
        >>> monitor.finish()
        >>> run.finish()
    """

    def __init__(self, run: Any | None = None) -> None:
        """Create a tracker.

        Args:
            run: The `wandb.Run` returned by `wandb.init()`. If omitted,
                `wandb.run` (the currently active run, if any) is used at
                logging time instead -- resolved lazily on each call
                rather than at construction, so this still works if
                `wandb.init()` hasn't been called yet when the tracker
                itself is constructed.

        Raises:
            ImportError: If the `wandb` package is not installed.
        """
        _require_wandb()
        super().__init__()
        self._run = run

    @property
    def _active_run(self) -> Any | None:
        return self._run if self._run is not None else wandb.run

    def _log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        run = self._active_run
        if run is None:
            raise RuntimeError(
                "WandbTracker has no active W&B run. Call wandb.init() first, "
                "or pass run=... explicitly."
            )
        run.log(metrics, step=step)

    def _log_summary(self, summary: dict[str, Any]) -> None:
        if not summary:
            return
        run = self._active_run
        if run is None:
            return
        run.summary.update(summary)
