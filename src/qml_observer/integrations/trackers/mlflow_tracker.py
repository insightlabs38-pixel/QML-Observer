"""MLflowTracker: write QMLMonitor output into an existing MLflow run.

Milestone 14 (`future_milestones_plan.md`), Issue #101 ("Experiment-tracker
integrations").

Requires the optional `mlflow` dependency (`pip install
qml-observer[mlflow]`). Does *not* start or end an MLflow run itself --
per the milestone's framing ("writing output into existing tracking
infra, not a competing tracker"), attach this to a run you've already
opened yourself (typically via `with mlflow.start_run(): ...` around your
training loop, or by passing an explicit `run_id`).
"""

from __future__ import annotations

from typing import Any

from qml_observer.integrations.trackers.base import BaseExperimentTracker

try:
    import mlflow
except ImportError as _exc:  # pragma: no cover - exercised only without mlflow installed
    mlflow = None  # type: ignore[assignment]
    _IMPORT_ERROR: ImportError | None = _exc
else:
    _IMPORT_ERROR = None


def _require_mlflow() -> None:
    if mlflow is None:
        raise ImportError(
            "MLflowTracker requires the optional 'mlflow' dependency. "
            "Install it with `pip install qml-observer[mlflow]` or "
            "`pip install mlflow>=2.0`."
        ) from _IMPORT_ERROR


class MLflowTracker(BaseExperimentTracker):
    """Forwards `QMLMonitor` events/diagnoses into MLflow as metrics and tags.

    Example (active-run form, no `run_id` needed):
        >>> import mlflow
        >>> from qml_observer import QMLMonitor
        >>> from qml_observer.integrations.trackers.mlflow_tracker import MLflowTracker
        >>>
        >>> with mlflow.start_run():
        ...     monitor = QMLMonitor(reporter=MLflowTracker())
        ...     for step in range(1000):
        ...         monitor.update(step=step, loss=loss)
        ...     monitor.finish()

    Example (explicit `run_id`, e.g. logging from a worker process that
    didn't itself call `mlflow.start_run()`):
        >>> tracker = MLflowTracker(run_id=run_id, tracking_uri="http://mlflow.internal:5000")
        >>> monitor = QMLMonitor(reporter=tracker)
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        """Create a tracker.

        Args:
            run_id: If given, metrics/tags are logged to this specific
                run via `mlflow.tracking.MlflowClient`, regardless of
                whether an MLflow run is "active" in this process. If
                omitted, the module-level `mlflow.log_metric`/`set_tags`
                calls are used instead, which require an active run
                (typically a surrounding `with mlflow.start_run():`).
            tracking_uri: If given, calls `mlflow.set_tracking_uri()`
                before anything else -- convenient when the tracker is
                constructed in a process that hasn't otherwise configured
                MLflow's tracking server.

        Raises:
            ImportError: If the `mlflow` package is not installed.
        """
        _require_mlflow()
        super().__init__()
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        self._run_id = run_id
        self._client = mlflow.tracking.MlflowClient() if run_id is not None else None

    def _log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if self._client is not None:
                assert self._run_id is not None
                self._client.log_metric(self._run_id, key, value, step=step)
            else:
                mlflow.log_metric(key, value, step=step)

    def _log_summary(self, summary: dict[str, Any]) -> None:
        if not summary:
            return
        tags = {f"qml_observer.{key}": str(value) for key, value in summary.items()}
        if self._client is not None:
            assert self._run_id is not None
            for key, value in tags.items():
                self._client.set_tag(self._run_id, key, value)
        else:
            mlflow.set_tags(tags)
