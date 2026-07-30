"""Unit tests for qml_observer.dashboard.export (Milestone 11, Issue #82)."""

from __future__ import annotations

import csv
import io
import json

import numpy as np

from qml_observer.core.monitor import QMLMonitor
from qml_observer.dashboard.data_source import MonitorDataSource, ReporterDataSource
from qml_observer.dashboard.export import build_export_payload, export_csv, export_json
from qml_observer.reporting.reporter import RunReporter


def _monitor_with_gradients(n_steps=5):
    monitor = QMLMonitor(planned_steps=50, run_id="run-export-test")
    rng = np.random.default_rng(0)
    for step in range(n_steps):
        monitor.update(step=step, loss=1.0 / (step + 1), gradients=rng.normal(0, 0.05, size=4))
    return monitor


class TestBuildExportPayload:
    def test_contains_all_expected_sections(self):
        source = MonitorDataSource(_monitor_with_gradients())
        payload = build_export_payload(source)
        assert set(payload.keys()) == {"run_id", "loss", "gradient", "diagnosis", "compute"}
        assert payload["run_id"] == "run-export-test"
        assert len(payload["loss"]["steps"]) == 5
        assert len(payload["gradient"]["steps"]) == 5


class TestExportJson:
    def test_is_valid_json_matching_payload(self):
        source = MonitorDataSource(_monitor_with_gradients())
        text = export_json(source)
        parsed = json.loads(text)
        assert parsed == build_export_payload(source)


class TestExportCsv:
    def test_header_and_row_count(self):
        source = MonitorDataSource(_monitor_with_gradients(n_steps=4))
        text = export_csv(source)
        rows = list(csv.reader(io.StringIO(text)))
        assert rows[0] == ["step", "loss", "gradient_norm_l2", "gradient_variance", "gradient_snr"]
        assert len(rows) == 5  # header + 4 steps

    def test_blank_not_zero_when_gradient_unavailable(self):
        reporter = RunReporter()
        monitor = QMLMonitor(reporter=reporter)
        rng = np.random.default_rng(0)
        for step in range(3):
            monitor.update(step=step, loss=1.0, gradients=rng.normal(size=4))
        monitor.finish()

        source = ReporterDataSource(reporter)
        text = export_csv(source)
        rows = list(csv.reader(io.StringIO(text)))
        assert rows[0][0] == "step"
        for row in rows[1:]:
            # gradient columns must be blank ("") not "0" when unavailable
            assert row[2] == ""
            assert row[3] == ""
            assert row[4] == ""

    def test_no_steps_produces_header_only(self):
        reporter = RunReporter()
        source = ReporterDataSource(reporter)
        text = export_csv(source)
        rows = list(csv.reader(io.StringIO(text)))
        assert len(rows) == 1
