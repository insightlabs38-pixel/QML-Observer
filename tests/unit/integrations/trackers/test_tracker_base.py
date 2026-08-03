"""Unit tests for qml_observer.integrations.trackers.base.

Milestone 14, Issue #101 ("Experiment-tracker integrations"). No optional
dependency required -- exercises the shared, dependency-free plumbing
`MLflowTracker`/`WandbTracker` both build on.
"""

from __future__ import annotations

import pytest

from qml_observer.integrations.trackers.base import (
    BaseExperimentTracker,
    diagnosis_metrics,
    event_metrics,
)
from qml_observer.schemas.training import TrainingEvent


def _event(step=0, loss=None, wall_time=None):
    return TrainingEvent(run_id="run-1", step=step, loss=loss, wall_time=wall_time)


class TestEventMetrics:
    def test_extracts_loss_and_wall_time(self):
        metrics = event_metrics(_event(step=3, loss=0.5, wall_time=1.2))
        assert metrics == {"loss": 0.5, "wall_time": 1.2}

    def test_omits_missing_fields(self):
        metrics = event_metrics(_event(step=3))
        assert metrics == {}

    def test_partial_fields(self):
        metrics = event_metrics(_event(step=3, loss=0.1))
        assert metrics == {"loss": 0.1}


class TestDiagnosisMetrics:
    def test_extracts_expected_fields(self, warning_diagnosis):
        metrics = diagnosis_metrics(warning_diagnosis)
        assert metrics["final_issue"] == warning_diagnosis.issue.value
        assert metrics["confidence"] == warning_diagnosis.confidence
        assert metrics["severity"] == warning_diagnosis.severity
        assert metrics["degraded"] is False


class _RecordingTracker(BaseExperimentTracker):
    """A concrete tracker recording every call, for testing the base skeleton."""

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.metric_calls = []
        self.summary_calls = []

    def _log_metrics(self, step, metrics):
        if self.fail:
            raise RuntimeError("boom")
        self.metric_calls.append((step, dict(metrics)))

    def _log_summary(self, summary):
        if self.fail:
            raise RuntimeError("boom")
        self.summary_calls.append(dict(summary))


class TestBaseExperimentTracker:
    def test_record_event_logs_metrics(self):
        tracker = _RecordingTracker()
        tracker.record_event(_event(step=5, loss=0.3))
        assert tracker.metric_calls == [(5, {"loss": 0.3})]

    def test_record_event_skips_when_no_metrics(self):
        tracker = _RecordingTracker()
        tracker.record_event(_event(step=5))
        assert tracker.metric_calls == []

    def test_record_diagnosis_logs_confidence(self, healthy_diagnosis):
        tracker = _RecordingTracker()
        tracker.record_event(_event(step=7))
        tracker.record_diagnosis(healthy_diagnosis)
        assert tracker.metric_calls == [(7, {"confidence": healthy_diagnosis.confidence})]

    def test_finalize_returns_and_logs_summary(self, critical_diagnosis):
        tracker = _RecordingTracker()
        tracker.record_diagnosis(critical_diagnosis)
        summary = tracker.finalize()
        assert summary["final_issue"] == critical_diagnosis.issue.value
        assert tracker.summary_calls == [summary]

    def test_finalize_with_no_diagnosis_yet(self):
        tracker = _RecordingTracker()
        assert tracker.finalize() == {}

    def test_record_event_is_fail_open(self):
        tracker = _RecordingTracker(fail=True)
        tracker.record_event(_event(step=1, loss=0.1))  # must not raise

    def test_record_diagnosis_is_fail_open(self, healthy_diagnosis):
        tracker = _RecordingTracker(fail=True)
        tracker.record_diagnosis(healthy_diagnosis)  # must not raise

    def test_finalize_is_fail_open(self, healthy_diagnosis):
        tracker = _RecordingTracker(fail=True)
        tracker.record_diagnosis(healthy_diagnosis)
        summary = tracker.finalize()  # must not raise
        assert summary["final_issue"] == healthy_diagnosis.issue.value

    def test_unimplemented_hooks_raise_not_implemented(self):
        tracker = BaseExperimentTracker()
        with pytest.raises(NotImplementedError):
            tracker._log_metrics(0, {})
        with pytest.raises(NotImplementedError):
            tracker._log_summary({})
