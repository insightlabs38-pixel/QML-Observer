"""Unit tests for qml_observer.integrations.trackers.mlflow_tracker.MLflowTracker.

Milestone 14, Issue #101. Skipped entirely if the optional `mlflow`
dependency isn't installed (`pip install qml-observer[mlflow]`). Uses a
local file-based tracking URI (no network) via `tmp_path`.
"""

from __future__ import annotations

import pytest

mlflow = pytest.importorskip("mlflow")

from qml_observer.integrations.trackers.mlflow_tracker import MLflowTracker  # noqa: E402
from qml_observer.schemas.training import TrainingEvent  # noqa: E402


@pytest.fixture(autouse=True)
def _local_tracking_uri(tmp_path, monkeypatch):
    """Point MLflow at a throwaway local sqlite store for every test in this module.

    (The plain filesystem tracking backend is in maintenance mode as of
    MLflow 3.x and refuses to initialize without an explicit opt-out env
    var -- sqlite is the modern equivalent of "just a local file", with no
    network involved either.)
    """
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    yield
    # Guard against a run left active by a previous test bleeding into the
    # next one (MLflow's "active run" is global fluent-API state, not
    # scoped to a tracking URI).
    while mlflow.active_run() is not None:
        mlflow.end_run()


def _event(step=0, loss=None, wall_time=None):
    return TrainingEvent(run_id="run-1", step=step, loss=loss, wall_time=wall_time)


class TestConstruction:
    def test_requires_active_or_explicit_run_by_default(self):
        # Construction itself never requires an active run -- only logging does.
        tracker = MLflowTracker()
        assert tracker._client is None

    def test_explicit_run_id_uses_client(self):
        with mlflow.start_run() as run:
            run_id = run.info.run_id
        tracker = MLflowTracker(run_id=run_id)
        assert tracker._client is not None
        assert tracker._run_id == run_id


class TestActiveRunLogging:
    def test_record_event_logs_metric_to_active_run(self):
        with mlflow.start_run() as run:
            tracker = MLflowTracker()
            tracker.record_event(_event(step=0, loss=0.5))
            client = mlflow.tracking.MlflowClient()
            history = client.get_metric_history(run.info.run_id, "loss")
        assert len(history) == 1
        assert history[0].value == pytest.approx(0.5)

    def test_finalize_sets_tags_on_active_run(self, healthy_diagnosis):
        with mlflow.start_run() as run:
            tracker = MLflowTracker()
            tracker.record_diagnosis(healthy_diagnosis)
            tracker.finalize()
            client = mlflow.tracking.MlflowClient()
            tags = client.get_run(run.info.run_id).data.tags
        assert tags.get("qml_observer.final_issue") == healthy_diagnosis.issue.value

    def test_no_active_run_is_fail_open(self):
        # No `with mlflow.start_run():` -- logging should fail internally
        # but never raise out of record_event (BaseExperimentTracker's
        # fail-open wrapper).
        tracker = MLflowTracker()
        tracker.record_event(_event(step=0, loss=0.5))


class TestExplicitRunId:
    def test_record_event_logs_metric_via_client(self):
        with mlflow.start_run() as run:
            run_id = run.info.run_id
        tracker = MLflowTracker(run_id=run_id)
        tracker.record_event(_event(step=0, loss=0.75))

        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(run_id, "loss")
        assert len(history) == 1
        assert history[0].value == pytest.approx(0.75)
