"""Unit tests for qml_observer.dashboard.data_source (Milestone 11, Issue #76).

Covers all three built-in `DashboardDataSource` implementations against
the same underlying run, so behavior parity/differences between them
(especially the gradient-availability difference) are pinned explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from qml_observer.core.monitor import QMLMonitor
from qml_observer.dashboard.data_source import (
    JSONLDataSource,
    MonitorDataSource,
    ReporterDataSource,
)
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.reporting.reporter import RunReporter


def _run_monitor_with_gradients(monitor: QMLMonitor, n_steps: int = 15) -> None:
    rng = np.random.default_rng(0)
    for step in range(n_steps):
        monitor.update(step=step, loss=1.0 / (step + 1), gradients=rng.normal(0, 0.05, size=6))


class TestMonitorDataSource:
    def test_loss_series_matches_window(self):
        monitor = QMLMonitor(planned_steps=100)
        _run_monitor_with_gradients(monitor, n_steps=10)
        source = MonitorDataSource(monitor)

        series = source.loss_series()
        assert series.steps == list(range(10))
        assert series.loss[0] == pytest.approx(1.0)
        assert series.loss[-1] == pytest.approx(0.1)

    def test_gradient_series_available_with_gradients(self):
        monitor = QMLMonitor()
        _run_monitor_with_gradients(monitor, n_steps=8)
        source = MonitorDataSource(monitor)

        series = source.gradient_series()
        assert series.available
        assert len(series.steps) == 8
        assert all(v is not None for v in series.norm_l2)

    def test_gradient_series_empty_without_gradients(self):
        monitor = QMLMonitor()
        for step in range(5):
            monitor.update(step=step, loss=1.0)
        source = MonitorDataSource(monitor)

        series = source.gradient_series()
        assert not series.available
        assert series.steps == []

    def test_diagnosis_reflects_latest(self):
        monitor = QMLMonitor(detectors=[BarrenPlateauDetector()])
        _run_monitor_with_gradients(monitor, n_steps=5)
        source = MonitorDataSource(monitor)

        diag = source.diagnosis()
        assert diag is not None
        assert diag["issue"] == monitor.latest_diagnosis().issue.value
        assert diag["degraded"] is False

    def test_diagnosis_none_before_any_step(self):
        monitor = QMLMonitor()
        source = MonitorDataSource(monitor)
        assert source.diagnosis() is None

    def test_compute_usage_uses_planned_steps(self):
        monitor = QMLMonitor(planned_steps=100)
        _run_monitor_with_gradients(monitor, n_steps=10)
        source = MonitorDataSource(monitor, framework="generic")

        usage = source.compute_usage()
        assert usage.actual_steps == 10
        assert usage.planned_steps == 100
        assert usage.framework == "generic"
        assert usage.estimated_compute_saved is not None
        assert usage.estimated_compute_saved >= 0

    def test_compute_usage_unknown_without_planned_steps(self):
        monitor = QMLMonitor()
        _run_monitor_with_gradients(monitor, n_steps=3)
        source = MonitorDataSource(monitor)

        usage = source.compute_usage()
        assert usage.estimated_compute_saved is None
        assert usage.formatted == "unknown (no planned_steps configured)"

    def test_run_id_matches_monitor(self):
        monitor = QMLMonitor(run_id="run-fixed-id")
        source = MonitorDataSource(monitor)
        assert source.run_id() == "run-fixed-id"


class TestReporterDataSource:
    def test_loss_series_from_events(self):
        reporter = RunReporter(planned_steps=20)
        monitor = QMLMonitor(reporter=reporter, planned_steps=20)
        for step in range(6):
            monitor.update(step=step, loss=float(step))
        monitor.finish()

        source = ReporterDataSource(reporter)
        series = source.loss_series()
        assert series.steps == list(range(6))
        assert series.loss == [float(s) for s in range(6)]

    def test_gradient_series_always_empty(self):
        reporter = RunReporter()
        monitor = QMLMonitor(reporter=reporter)
        rng = np.random.default_rng(1)
        for step in range(4):
            monitor.update(step=step, loss=1.0, gradients=rng.normal(size=4))
        monitor.finish()

        source = ReporterDataSource(reporter)
        series = source.gradient_series()
        assert not series.available
        assert series.to_dict() == {"steps": [], "norm_l2": [], "variance": [], "snr": []}

    def test_diagnosis_none_before_finish(self):
        reporter = RunReporter()
        source = ReporterDataSource(reporter)
        assert source.diagnosis() is None

    def test_compute_usage_after_finalize(self):
        reporter = RunReporter(framework="pennylane", planned_steps=50)
        monitor = QMLMonitor(reporter=reporter, planned_steps=50)
        for step in range(5):
            monitor.update(step=step, loss=1.0)
        monitor.finish()

        source = ReporterDataSource(reporter, framework="pennylane")
        usage = source.compute_usage()
        assert usage.actual_steps == 5
        assert usage.framework == "pennylane"
        assert usage.estimated_compute_saved == reporter.summary["estimated_compute_saved"]


class TestJSONLDataSource:
    def test_reads_loss_and_diagnosis_from_log(self, tmp_path):
        path = tmp_path / "run.jsonl"
        reporter = RunReporter(path, framework="generic", planned_steps=30)
        monitor = QMLMonitor(reporter=reporter, planned_steps=30)
        for step in range(7):
            monitor.update(step=step, loss=1.0 / (step + 1))
        monitor.finish()

        source = JSONLDataSource(path)
        loss = source.loss_series()
        assert loss.steps == list(range(7))

        diag = source.diagnosis()
        assert diag is not None
        assert diag["issue"] == "insufficient_evidence"

        usage = source.compute_usage()
        assert usage.actual_steps == 7
        assert usage.estimated_compute_saved == reporter.summary["estimated_compute_saved"]

    def test_missing_file_returns_empty_data_not_error(self, tmp_path):
        source = JSONLDataSource(tmp_path / "does-not-exist.jsonl")
        assert source.loss_series().to_dict() == {"steps": [], "loss": []}
        assert source.diagnosis() is None
        assert source.run_id() is None

    def test_gradient_series_present_when_logged_manually(self, tmp_path):
        from qml_observer.reporting.jsonl import JSONLWriter, event_record
        from qml_observer.schemas.gradient import summarize_gradient
        from qml_observer.schemas.training import TrainingEvent

        path = tmp_path / "run.jsonl"
        with JSONLWriter(path) as writer:
            for step in range(3):
                event = TrainingEvent(run_id="r1", step=step, loss=1.0)
                gradient = summarize_gradient(np.array([0.1, 0.2, 0.05]))
                writer.write(event_record(event, gradient=gradient))

        source = JSONLDataSource(path)
        series = source.gradient_series()
        assert series.available
        assert len(series.steps) == 3
