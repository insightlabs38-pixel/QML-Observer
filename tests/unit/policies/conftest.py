"""Shared fixtures for actions/policy unit tests."""

from __future__ import annotations

import pytest

from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType


def make_diagnosis(
    issue: IssueType = IssueType.HEALTHY,
    confidence: float = 0.9,
    severity: str = "info",
    evidence=None,
    recommendations=None,
    degraded: bool = False,
    degraded_reason: str | None = None,
) -> DiagnosisResult:
    """Build a `DiagnosisResult` with only the fields a test cares about."""
    return DiagnosisResult(
        issue=issue,
        confidence=confidence,
        severity=severity,
        evidence=evidence if evidence is not None else [],
        recommendations=recommendations if recommendations is not None else [],
        degraded=degraded,
        degraded_reason=degraded_reason,
    )


@pytest.fixture
def diagnosis_factory():
    return make_diagnosis


@pytest.fixture
def healthy_diagnosis():
    return make_diagnosis(issue=IssueType.HEALTHY, confidence=0.95, severity="info")


@pytest.fixture
def warning_diagnosis():
    return make_diagnosis(
        issue=IssueType.STAGNATION,
        confidence=0.6,
        severity="warning",
        evidence=["loss flat for 120 steps"],
        recommendations=["Consider adjusting the learning rate"],
    )


@pytest.fixture
def critical_diagnosis():
    return make_diagnosis(
        issue=IssueType.POSSIBLE_BARREN_PLATEAU,
        confidence=0.92,
        severity="critical",
        evidence=["gradient norm 2.4e-9 for 240 steps"],
        recommendations=["Consider reinitializing parameters"],
    )


@pytest.fixture
def degraded_critical_diagnosis():
    return make_diagnosis(
        issue=IssueType.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        severity="critical",
        degraded=True,
        degraded_reason="BarrenPlateauDetector raised ZeroDivisionError at step 42",
    )
