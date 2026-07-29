"""Unit tests for qml_observer.integrations.payloads."""

from __future__ import annotations

from qml_observer.integrations.payloads import SEVERITY_RANK, build_alert_payload, redact_payload
from qml_observer.schemas.diagnosis import SEVERITY_LEVELS


class TestSeverityRank:
    def test_matches_severity_levels_vocabulary(self):
        assert set(SEVERITY_RANK) == SEVERITY_LEVELS

    def test_ordering_is_monotonic(self):
        assert SEVERITY_RANK["info"] < SEVERITY_RANK["warning"] < SEVERITY_RANK["critical"]


class TestBuildAlertPayload:
    def test_basic_fields_from_diagnosis(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis)
        assert payload.severity == "critical"
        assert payload.issue == "possible_barren_plateau"
        assert payload.confidence == critical_diagnosis.confidence
        assert payload.evidence == critical_diagnosis.evidence
        assert payload.recommendations == critical_diagnosis.recommendations
        assert payload.degraded is False
        assert payload.run_id is None
        assert payload.current_metrics == {}

    def test_optional_run_id_and_metrics(self, warning_diagnosis):
        payload = build_alert_payload(
            warning_diagnosis,
            run_id="run-123",
            current_metrics={"step": 42, "loss": 0.5},
        )
        assert payload.run_id == "run-123"
        assert payload.current_metrics == {"step": 42, "loss": 0.5}

    def test_degraded_diagnosis_propagates_flag(self, degraded_critical_diagnosis):
        payload = build_alert_payload(degraded_critical_diagnosis)
        assert payload.degraded is True

    def test_to_dict_is_json_serializable_shape(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis, run_id="run-1")
        d = payload.to_dict()
        assert d["run_id"] == "run-1"
        assert d["severity"] == "critical"
        assert d["issue"] == "possible_barren_plateau"
        assert isinstance(d["evidence"], list)
        assert isinstance(d["recommendations"], list)
        assert "timestamp" in d

    def test_mutating_returned_lists_does_not_affect_source_diagnosis(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis)
        payload.evidence.append("mutated")
        assert "mutated" not in critical_diagnosis.evidence


class TestRedactPayload:
    def test_strips_evidence_and_metrics(self, critical_diagnosis):
        payload = build_alert_payload(
            critical_diagnosis, run_id="run-1", current_metrics={"step": 42}
        )
        redacted = redact_payload(payload)
        assert redacted.evidence == []
        assert redacted.current_metrics == {}
        assert redacted.redacted is True

    def test_preserves_summary_fields(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis, run_id="run-1")
        redacted = redact_payload(payload)
        assert redacted.severity == payload.severity
        assert redacted.issue == payload.issue
        assert redacted.confidence == payload.confidence
        assert redacted.run_id == payload.run_id
        assert redacted.degraded == payload.degraded
        assert redacted.recommendations == payload.recommendations

    def test_original_payload_is_unmodified(self, critical_diagnosis):
        payload = build_alert_payload(critical_diagnosis, current_metrics={"step": 1})
        redact_payload(payload)
        assert payload.evidence != []
        assert payload.current_metrics == {"step": 1}
        assert payload.redacted is False
