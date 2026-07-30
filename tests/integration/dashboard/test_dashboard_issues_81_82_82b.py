"""Integration tests for Milestone 11 Issues #81, #82, #82b."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from qml_observer.core.monitor import QMLMonitor
from qml_observer.dashboard import MonitorDataSource, create_app, run_dashboard
from qml_observer.reporting.reporter import RunReporter


def _finalize_run(path, n_steps=5):
    reporter = RunReporter(path, framework="pennylane", planned_steps=20)
    monitor = QMLMonitor(reporter=reporter, planned_steps=20)
    for step in range(n_steps):
        monitor.update(step=step, loss=1.0 / (step + 1))
    monitor.finish()


@pytest.fixture
def live_monitor() -> QMLMonitor:
    monitor = QMLMonitor(planned_steps=40, run_id="run-dashboard-it2")
    rng = np.random.default_rng(7)
    for step in range(6):
        monitor.update(step=step, loss=1.0 / (step + 1), gradients=rng.normal(0, 0.05, size=4))
    return monitor


class TestHistoryRoute:
    def test_unconfigured_returns_null_directory(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        body = client.get("/api/history").json()
        assert body == {"directory": None, "entries": []}

    def test_configured_directory_lists_finalized_runs(self, live_monitor, tmp_path):
        _finalize_run(tmp_path / "run_a.jsonl")
        _finalize_run(tmp_path / "run_b.jsonl", n_steps=8)

        client = TestClient(create_app(MonitorDataSource(live_monitor), history_dir=tmp_path))
        body = client.get("/api/history").json()
        assert body["directory"] == str(tmp_path)
        assert len(body["entries"]) == 2
        assert {e["steps"] for e in body["entries"]} == {5, 8}

    def test_empty_directory_returns_empty_entries(self, live_monitor, tmp_path):
        client = TestClient(create_app(MonitorDataSource(live_monitor), history_dir=tmp_path))
        body = client.get("/api/history").json()
        assert body["directory"] == str(tmp_path)
        assert body["entries"] == []


class TestExportRoutes:
    def test_export_json_downloadable_and_matches_api(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        response = client.get("/api/export.json")
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        assert "run-dashboard-it2.json" in response.headers["content-disposition"]

        payload = json.loads(response.text)
        assert payload["loss"] == client.get("/api/loss").json()
        assert payload["run_id"] == "run-dashboard-it2"

    def test_export_csv_downloadable_and_well_formed(self, live_monitor):
        client = TestClient(create_app(MonitorDataSource(live_monitor)))
        response = client.get("/api/export.csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "run-dashboard-it2.csv" in response.headers["content-disposition"]

        rows = list(csv.reader(io.StringIO(response.text)))
        assert rows[0][0] == "step"
        assert len(rows) == 7  # header + 6 steps


class TestNonLoopbackRefusal:
    def test_default_host_does_not_raise_before_reaching_uvicorn(self, monkeypatch, live_monitor):
        # We don't want to actually bind a port in the test suite; patch
        # uvicorn.run so this only checks that the safeguard doesn't fire
        # for the (safe) default host.
        called = {}

        def fake_run(app, *, host, port, log_level):
            called["host"] = host

        import uvicorn

        monkeypatch.setattr(uvicorn, "run", fake_run)

        run_dashboard(MonitorDataSource(live_monitor))
        assert called["host"] == "127.0.0.1"

    def test_non_loopback_host_raises_without_opt_in(self, live_monitor):
        with pytest.raises(ValueError, match="refusing to bind"):
            run_dashboard(MonitorDataSource(live_monitor), host="0.0.0.0")

    def test_non_loopback_host_allowed_with_explicit_opt_in(self, monkeypatch, live_monitor):
        import uvicorn

        called = {}

        def fake_run(app, *, host, port, log_level):
            called["host"] = host

        monkeypatch.setattr(uvicorn, "run", fake_run)

        run_dashboard(MonitorDataSource(live_monitor), host="0.0.0.0", allow_non_loopback=True)
        assert called["host"] == "0.0.0.0"
