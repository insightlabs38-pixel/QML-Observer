"""Integration tests for QMLMonitor's optional telemetry wiring
(addendum §5). Lives alongside core/monitor.py's own tests since it
exercises `_maybe_emit_telemetry`, not the telemetry package in
isolation (see tests/unit/telemetry/ for that).
"""

import json

import pytest

from qml_observer.core.monitor import QMLMonitor
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.telemetry import consent
from qml_observer.telemetry.collector import TelemetryCollector


@pytest.fixture
def consent_path(tmp_path):
    return tmp_path / "telemetry.json"


@pytest.fixture(autouse=True)
def _patch_consent_path(monkeypatch, consent_path):
    monkeypatch.setattr(
        "qml_observer.telemetry.collector.is_enabled",
        lambda: consent.is_enabled(consent_path),
    )


class TestMonitorTelemetryOptIn:
    def test_no_collector_means_no_telemetry_code_runs(self, tmp_path):
        # No telemetry_collector passed at all -- must not touch the
        # telemetry package or raise, regardless of consent state.
        monitor = QMLMonitor(detectors=[BarrenPlateauDetector()])
        monitor.update(step=0, loss=1.0, gradients=[0.1, 0.1])
        monitor.finish()  # must not raise

    def test_collector_given_but_not_enabled_collects_nothing(self, tmp_path, consent_path):
        queue_path = tmp_path / "queue.jsonl"
        collector = TelemetryCollector(queue_path=queue_path)
        monitor = QMLMonitor(detectors=[BarrenPlateauDetector()], telemetry_collector=collector)
        monitor.update(step=0, loss=1.0, gradients=[0.1, 0.1])
        monitor.finish()
        assert not queue_path.exists()

    def test_enabled_collector_receives_record_on_finish(self, tmp_path, consent_path):
        consent.enable(consent_path)
        queue_path = tmp_path / "queue.jsonl"
        collector = TelemetryCollector(queue_path=queue_path)
        monitor = QMLMonitor(
            detectors=[BarrenPlateauDetector()],
            telemetry_collector=collector,
            telemetry_framework="pennylane",
        )
        monitor.update(step=0, loss=1.0, gradients=[0.1, 0.1])
        monitor.finish()

        lines = queue_path.read_text().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["framework"] == "pennylane"
        assert "BarrenPlateauDetector" in payload["detector_names"]
        assert payload["thresholds"]  # non-empty: real thresholds extracted

    def test_finish_never_raises_if_telemetry_collector_is_broken(self, tmp_path, consent_path):
        consent.enable(consent_path)

        class BrokenCollector:
            def maybe_collect(self, record):
                raise RuntimeError("boom")

        monitor = QMLMonitor(
            detectors=[BarrenPlateauDetector()], telemetry_collector=BrokenCollector()
        )
        monitor.update(step=0, loss=1.0, gradients=[0.1, 0.1])
        # Fail-open (addendum §1): a broken telemetry collector must
        # never break finish() or the caller's training loop.
        diagnosis = monitor.finish()
        assert diagnosis is not None
