"""Unit tests for qml_observer.schemas.diagnosis."""

import json

import pytest

from qml_observer.schemas.diagnosis import SEVERITY_LEVELS, DiagnosisResult, IssueType


class TestIssueType:
    def test_all_expected_members_exist(self):
        expected = {
            "HEALTHY",
            "CONVERGED",
            "POSSIBLE_BARREN_PLATEAU",
            "STAGNATION",
            "NOISE_DOMINATED",
            "UNSTABLE",
            "INSUFFICIENT_EVIDENCE",
        }
        assert {member.name for member in IssueType} == expected

    def test_behaves_as_a_string(self):
        assert IssueType.HEALTHY == "healthy"
        assert str(IssueType.HEALTHY) == "healthy"

    def test_json_serializes_as_plain_string(self):
        payload = json.dumps({"issue": IssueType.POSSIBLE_BARREN_PLATEAU})
        assert json.loads(payload)["issue"] == "possible_barren_plateau"


class TestDiagnosisResultConstruction:
    def _kwargs(self, **overrides):
        base = dict(
            issue=IssueType.HEALTHY,
            confidence=0.9,
            severity="info",
            evidence=[],
            recommendations=[],
        )
        base.update(overrides)
        return base

    def test_minimal_valid_construction(self):
        result = DiagnosisResult(**self._kwargs())
        assert result.degraded is False
        assert result.degraded_reason is None

    def test_full_construction_with_evidence(self):
        result = DiagnosisResult(
            **self._kwargs(
                issue=IssueType.POSSIBLE_BARREN_PLATEAU,
                confidence=0.91,
                severity="critical",
                evidence=["gradient norm 2.4e-9 for 240 steps"],
                recommendations=["Consider reinitializing parameters"],
            )
        )
        assert result.issue == IssueType.POSSIBLE_BARREN_PLATEAU

    @pytest.mark.parametrize("severity", sorted(SEVERITY_LEVELS))
    def test_all_severity_levels_are_valid(self, severity):
        DiagnosisResult(**self._kwargs(severity=severity))

    def test_degraded_result_requires_reason(self):
        result = DiagnosisResult(
            **self._kwargs(
                issue=IssueType.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                severity="warning",
                degraded=True,
                degraded_reason="BarrenPlateauDetector raised ZeroDivisionError at step 42",
            )
        )
        assert result.degraded is True


class TestDiagnosisResultValidation:
    def _kwargs(self, **overrides):
        base = dict(
            issue=IssueType.HEALTHY,
            confidence=0.9,
            severity="info",
            evidence=[],
            recommendations=[],
        )
        base.update(overrides)
        return base

    def test_non_issue_type_raises(self):
        """A plain str that happens to match a member's value is still not
        an IssueType instance and must be rejected."""
        with pytest.raises(TypeError, match="issue"):
            DiagnosisResult(**self._kwargs(issue="healthy"))

    @pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0, -5])
    def test_confidence_out_of_range_raises(self, confidence):
        with pytest.raises(ValueError, match="confidence"):
            DiagnosisResult(**self._kwargs(confidence=confidence))

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_confidence_boundaries_are_inclusive(self, confidence):
        DiagnosisResult(**self._kwargs(confidence=confidence))

    def test_invalid_severity_raises(self):
        with pytest.raises(ValueError, match="severity"):
            DiagnosisResult(**self._kwargs(severity="bogus"))

    def test_non_str_evidence_item_raises(self):
        with pytest.raises(TypeError, match="evidence"):
            DiagnosisResult(**self._kwargs(evidence=["ok", 5]))

    def test_non_str_recommendation_item_raises(self):
        with pytest.raises(TypeError, match="recommendations"):
            DiagnosisResult(**self._kwargs(recommendations=[1]))

    def test_degraded_true_without_reason_raises(self):
        with pytest.raises(ValueError, match="degraded_reason is required"):
            DiagnosisResult(**self._kwargs(degraded=True))

    def test_degraded_false_with_reason_raises(self):
        with pytest.raises(ValueError, match="must be None when degraded=False"):
            DiagnosisResult(**self._kwargs(degraded=False, degraded_reason="oops"))
