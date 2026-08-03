"""Unit tests for qml_observer.integrations.trackers.wandb_tracker.WandbTracker.

Milestone 14, Issue #101. Skipped entirely if the optional `wandb`
dependency isn't installed (`pip install qml-observer[wandb]`). Runs in
`WANDB_MODE=offline` throughout -- no network access, writes only to a
local run directory.
"""

from __future__ import annotations

import pytest

wandb = pytest.importorskip("wandb")

from qml_observer.integrations.trackers.wandb_tracker import WandbTracker  # noqa: E402
from qml_observer.schemas.training import TrainingEvent  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    yield
    if wandb.run is not None:
        wandb.finish()


def _event(step=0, loss=None, wall_time=None):
    return TrainingEvent(run_id="run-1", step=step, loss=loss, wall_time=wall_time)


class TestConstruction:
    def test_construction_without_run_defers_resolution(self):
        tracker = WandbTracker()
        assert tracker._run is None


class TestExplicitRun:
    def test_record_event_logs_metric_to_explicit_run(self):
        run = wandb.init(project="qml-observer-tests")
        tracker = WandbTracker(run=run)
        tracker.record_event(_event(step=0, loss=0.4))
        run.finish()

    def test_finalize_updates_run_summary(self, healthy_diagnosis):
        run = wandb.init(project="qml-observer-tests")
        tracker = WandbTracker(run=run)
        tracker.record_diagnosis(healthy_diagnosis)
        summary = tracker.finalize()
        assert run.summary["final_issue"] == healthy_diagnosis.issue.value
        assert summary["final_issue"] == healthy_diagnosis.issue.value
        run.finish()


class TestActiveRunFallback:
    def test_uses_wandb_run_when_no_explicit_run_given(self):
        run = wandb.init(project="qml-observer-tests")
        tracker = WandbTracker()
        tracker.record_event(_event(step=0, loss=0.2))  # must not raise
        run.finish()


class TestNoActiveRun:
    def test_record_event_is_fail_open_without_any_run(self):
        tracker = WandbTracker()
        tracker.record_event(_event(step=0, loss=0.2))  # must not raise

    def test_finalize_is_fail_open_without_any_run(self, healthy_diagnosis):
        tracker = WandbTracker()
        tracker.record_diagnosis(healthy_diagnosis)
        tracker.finalize()  # must not raise
