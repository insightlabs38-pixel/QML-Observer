"""Unit tests for qml_observer.telemetry.collector."""

import json

import pytest

from qml_observer.telemetry import consent
from qml_observer.telemetry.collector import TelemetryCollector
from qml_observer.telemetry.schema import build_telemetry_record


@pytest.fixture
def consent_path(tmp_path):
    return tmp_path / "telemetry.json"


@pytest.fixture(autouse=True)
def _patch_consent_path(monkeypatch, consent_path):
    """Point the module-level default consent path at a temp file so
    tests never touch the real ~/.config/qml-observer/telemetry.json."""
    monkeypatch.setattr(
        "qml_observer.telemetry.collector.is_enabled",
        lambda: consent.is_enabled(consent_path),
    )


@pytest.fixture
def record():
    return build_telemetry_record(
        detector_names=["BarrenPlateauDetector"],
        thresholds={"BarrenPlateauDetector.gradient_threshold": 5e-6},
        issue="possible_barren_plateau",
        confidence=0.9,
    )


class TestDisabledByDefault:
    def test_no_op_when_not_enabled(self, record, tmp_path):
        collector = TelemetryCollector(queue_path=tmp_path / "queue.jsonl")
        collected = collector.maybe_collect(record)
        assert collected is False
        assert not (tmp_path / "queue.jsonl").exists()


class TestLocalQueueing:
    def test_queues_locally_when_enabled_and_no_endpoint(self, record, tmp_path, consent_path):
        consent.enable(consent_path)
        queue_path = tmp_path / "queue.jsonl"
        collector = TelemetryCollector(queue_path=queue_path)
        collected = collector.maybe_collect(record)
        assert collected is True
        lines = queue_path.read_text().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["issue"] == "possible_barren_plateau"

    def test_appends_multiple_records(self, record, tmp_path, consent_path):
        consent.enable(consent_path)
        queue_path = tmp_path / "queue.jsonl"
        collector = TelemetryCollector(queue_path=queue_path)
        collector.maybe_collect(record)
        collector.maybe_collect(record)
        assert len(queue_path.read_text().splitlines()) == 2

    def test_immediately_respects_disable_mid_session(self, record, tmp_path, consent_path):
        consent.enable(consent_path)
        queue_path = tmp_path / "queue.jsonl"
        collector = TelemetryCollector(queue_path=queue_path)
        collector.maybe_collect(record)
        consent.disable(consent_path)
        collector.maybe_collect(record)
        assert len(queue_path.read_text().splitlines()) == 1


class TestEndpointSend:
    def test_sends_to_endpoint_when_configured(self, record, monkeypatch, consent_path):
        consent.enable(consent_path)
        sent = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout):
            sent["url"] = request.full_url
            sent["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        monkeypatch.setattr("qml_observer.telemetry.collector.urllib.request.urlopen", fake_urlopen)
        collector = TelemetryCollector(endpoint="https://example.invalid/telemetry")
        collected = collector.maybe_collect(record)
        assert collected is True
        assert sent["url"] == "https://example.invalid/telemetry"
        assert sent["body"]["issue"] == "possible_barren_plateau"

    def test_send_failure_is_swallowed_and_logged(self, record, monkeypatch, consent_path):
        consent.enable(consent_path)

        def raising_urlopen(*_args, **_kwargs):
            raise OSError("network unreachable")

        monkeypatch.setattr(
            "qml_observer.telemetry.collector.urllib.request.urlopen", raising_urlopen
        )
        collector = TelemetryCollector(endpoint="https://example.invalid/telemetry")
        # Must never raise -- fail-open, addendum §1.
        collected = collector.maybe_collect(record)
        assert collected is False
