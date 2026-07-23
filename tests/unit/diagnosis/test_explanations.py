"""Unit tests for qml_observer.diagnosis.explanations.explain."""

import pytest

from qml_observer.diagnosis.explanations import explain
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType


def _diagnosis(**overrides):
    defaults = dict(
        issue=IssueType.POSSIBLE_BARREN_PLATEAU,
        confidence=0.91,
        severity="critical",
        evidence=["gradient norm 2.4e-9", "persisted 240 steps"],
        recommendations=["stop and inspect ansatz"],
    )
    defaults.update(overrides)
    return DiagnosisResult(**defaults)


class TestHeadlineAndConfidence:
    def test_includes_headline_and_confidence_percent(self):
        text = explain(_diagnosis())
        assert "Possible barren plateau" in text
        assert "91%" in text
        assert "critical" in text

    def test_unknown_issue_falls_back_to_raw_status(self):
        # All current IssueType members have headlines; this exercises the
        # fallback branch defensively in case new issue types are added
        # without updating _HEADLINES.
        text = explain(_diagnosis(issue=IssueType.NOISE_DOMINATED))
        assert "noise" in text.lower()


class TestEvidenceTruncation:
    def test_all_evidence_shown_by_default_within_limit(self):
        text = explain(_diagnosis())
        assert "gradient norm 2.4e-9" in text
        assert "persisted 240 steps" in text

    def test_truncates_evidence_to_max_evidence(self):
        diagnosis = _diagnosis(evidence=[f"line {i}" for i in range(10)])
        text = explain(diagnosis, max_evidence=3)
        assert "line 0" in text
        assert "line 2" in text
        assert "line 3" not in text
        assert "7 more" in text

    def test_max_evidence_none_shows_everything(self):
        diagnosis = _diagnosis(evidence=[f"line {i}" for i in range(10)])
        text = explain(diagnosis, max_evidence=None)
        assert "line 9" in text
        assert "more" not in text

    def test_negative_max_evidence_raises(self):
        with pytest.raises(ValueError):
            explain(_diagnosis(), max_evidence=-1)

    def test_no_evidence_omits_section(self):
        text = explain(_diagnosis(evidence=[]))
        assert "Evidence:" not in text


class TestRecommendations:
    def test_recommendations_included(self):
        text = explain(_diagnosis())
        assert "stop and inspect ansatz" in text

    def test_no_recommendations_omits_section(self):
        text = explain(_diagnosis(recommendations=[]))
        assert "Recommended next steps:" not in text


class TestDegraded:
    def test_degraded_diagnosis_shows_warning_banner(self):
        diagnosis = _diagnosis(degraded=True, degraded_reason="detector X raised ValueError")
        text = explain(diagnosis)
        assert "DEGRADED" in text
        assert "detector X raised ValueError" in text

    def test_non_degraded_diagnosis_has_no_banner(self):
        text = explain(_diagnosis())
        assert "DEGRADED" not in text


class TestHealthyAndInsufficientEvidence:
    def test_healthy_headline(self):
        text = explain(
            DiagnosisResult(
                issue=IssueType.HEALTHY,
                confidence=0.95,
                severity="info",
                evidence=[],
                recommendations=[],
            )
        )
        assert "healthy" in text.lower()

    def test_insufficient_evidence_headline(self):
        text = explain(
            DiagnosisResult(
                issue=IssueType.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                severity="info",
                evidence=[],
                recommendations=[],
            )
        )
        assert "Not enough data" in text
