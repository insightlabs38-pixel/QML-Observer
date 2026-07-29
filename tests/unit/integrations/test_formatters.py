"""Unit tests for qml_observer.integrations.formatters."""

from __future__ import annotations

from qml_observer.integrations.formatters import default_formatter, slack_formatter
from qml_observer.integrations.payloads import build_alert_payload


class TestDefaultFormatter:
    def test_returns_payload_to_dict(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis, run_id="run-1")
        assert default_formatter(payload) == payload.to_dict()


class TestSlackFormatter:
    def test_has_text_and_attachments(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis, run_id="run-1")
        body = slack_formatter(payload)
        assert "text" in body
        assert "qml-observer alert" in body["text"]
        assert len(body["attachments"]) == 1

    def test_critical_uses_danger_color(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis)
        body = slack_formatter(payload)
        assert body["attachments"][0]["color"] == "#E01E5A"

    def test_warning_uses_warning_color(self, warning_diagnosis):
        payload = build_alert_payload(warning_diagnosis)
        body = slack_formatter(payload)
        assert body["attachments"][0]["color"] == "#ECB22E"

    def test_includes_run_id_field_when_present(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis, run_id="run-abc")
        body = slack_formatter(payload)
        fields = body["attachments"][0]["fields"]
        assert any(f["title"] == "Run ID" and f["value"] == "run-abc" for f in fields)

    def test_omits_run_id_field_when_absent(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis)
        body = slack_formatter(payload)
        fields = body["attachments"][0]["fields"]
        assert not any(f["title"] == "Run ID" for f in fields)

    def test_includes_current_metrics_field(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis, current_metrics={"step": 42})
        body = slack_formatter(payload)
        fields = body["attachments"][0]["fields"]
        assert any(f["title"] == "Current metrics" and "step=42" in f["value"] for f in fields)

    def test_includes_evidence_and_recommendations(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis)
        body = slack_formatter(payload)
        fields = body["attachments"][0]["fields"]
        titles = {f["title"] for f in fields}
        assert "Evidence" in titles
        assert "Recommended action" in titles

    def test_degraded_marker_in_text(self, degraded_critical_diagnosis):
        payload = build_alert_payload(degraded_critical_diagnosis)
        body = slack_formatter(payload)
        assert "DEGRADED" in body["text"]

    def test_unknown_severity_falls_back_to_gray(self):
        # Guard against a future severity value not yet added to the color map.
        from qml_observer.integrations.payloads import AlertPayload

        payload = AlertPayload(severity="unknown", issue="healthy", confidence=0.5)
        body = slack_formatter(payload)
        assert body["attachments"][0]["color"] == "#999999"

    def test_redacted_payload_shows_withheld_note_not_raw_evidence(self, critical_diagnosis):
        from qml_observer.integrations.payloads import redact_payload

        payload = redact_payload(build_alert_payload(critical_diagnosis))
        body = slack_formatter(payload)
        fields = body["attachments"][0]["fields"]
        withheld_fields = [f for f in fields if f["title"] == "Evidence / metrics"]
        assert any("redacted" in f["value"].lower() for f in withheld_fields)
        assert not any(f["title"] == "Evidence" for f in fields)
