"""Integration tests for the dashboard FastAPI app (Milestone 11).

Exercises `create_app()` end-to-end via `TestClient` against all three
`DashboardDataSource` implementations, so the routes (Issues #77-#80) are
tested against real, if small, run data rather than mocks.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from qml_observer.core.monitor import QMLMonitor
from qml_observer.dashboard import (
    JSONLDataSource,
    MonitorDataSource,
    ReporterDataSource,
    create_app,
)
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.reporting.reporter import RunReporter


@pytest.fixture
def live_monitor() -> QMLMonitor:
    monitor = QMLMonitor(
        detectors=[BarrenPlateauDetector()],
        policy="warn",
        planned_steps=40,
        run_id="run-dashboard-it",
    )
    rng = np.random.default_rng(42)
    for step in range(12):
        monitor.update(step=step, loss=1.0 / (step + 1), gradients=rng.normal(0, 0.05, size=6))
    return monitor


class TestAppWithMonitorSource:
    def test_index_serves_html(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        response = client.get("/")
        assert response.status_code == 200
        assert "QML Observer Dashboard" in response.text

    def test_static_assets_served(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        for asset in ("/static/style.css", "/static/app.js", "/static/chart.js"):
            response = client.get(asset)
            assert response.status_code == 200, asset

    def test_status_reports_run_id(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        response = client.get("/api/status")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["run_id"] == "run-dashboard-it"

    def test_loss_endpoint_shape(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        body = client.get("/api/loss").json()
        assert body["steps"] == list(range(12))
        assert len(body["loss"]) == 12

    def test_gradient_endpoint_available_from_live_monitor(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        body = client.get("/api/gradient").json()
        assert body["available"] is True
        assert len(body["steps"]) == 12
        assert len(body["norm_l2"]) == 12

    def test_diagnosis_endpoint_reflects_latest_result(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        body = client.get("/api/diagnosis").json()
        assert body["issue"] == live_monitor.latest_diagnosis().issue.value
        assert body["degraded"] is False

    def test_compute_endpoint_uses_planned_steps(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor, framework="generic")))
        body = client.get("/api/compute").json()
        assert body["actual_steps"] == 12
        assert body["planned_steps"] == 40
        assert body["framework"] == "generic"
        assert "formatted" in body


class TestAppWithReporterSource:
    def test_gradient_endpoint_reports_unavailable(self):
        reporter = RunReporter()
        monitor = QMLMonitor(reporter=reporter)
        rng = np.random.default_rng(0)
        for step in range(5):
            monitor.update(step=step, loss=1.0, gradients=rng.normal(size=4))
        monitor.finish()

        client = TestClient(create_app(ReporterDataSource(reporter)))
        body = client.get("/api/gradient").json()
        assert body["available"] is False
        assert body["steps"] == []

    def test_diagnosis_endpoint_empty_before_finish(self):
        reporter = RunReporter()
        client = TestClient(create_app(ReporterDataSource(reporter)))
        body = client.get("/api/diagnosis").json()
        assert body == {}


class TestAppWithJSONLSource:
    def test_full_round_trip_from_log_file(self, tmp_path):
        path = tmp_path / "run.jsonl"
        reporter = RunReporter(path, framework="pennylane", planned_steps=25)
        monitor = QMLMonitor(reporter=reporter, planned_steps=25)
        for step in range(9):
            monitor.update(step=step, loss=1.0 / (step + 1))
        monitor.finish()

        client = TestClient(create_app(JSONLDataSource(path)))

        loss_body = client.get("/api/loss").json()
        assert loss_body["steps"] == list(range(9))

        compute_body = client.get("/api/compute").json()
        assert compute_body["actual_steps"] == 9
        assert (
            compute_body["estimated_compute_saved"]
            == reporter.summary["estimated_compute_saved"]
        )

    def test_missing_log_file_returns_empty_but_valid_responses(self, tmp_path):
        client = TestClient(create_app(JSONLDataSource(tmp_path / "missing.jsonl")))
        assert client.get("/api/loss").json() == {"steps": [], "loss": []}
        assert client.get("/api/diagnosis").json() == {}
        assert client.get("/api/status").json()["run_id"] is None
