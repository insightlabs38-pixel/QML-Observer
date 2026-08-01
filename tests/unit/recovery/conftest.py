"""Shared fixtures for qml_observer.recovery unit tests."""

from __future__ import annotations

import pytest

from qml_observer.recovery.base import RecoveryContext
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType
from qml_observer.schemas.gradient import GradientSnapshot
from qml_observer.schemas.optimizer import OptimizerMetadata


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
def barren_plateau_diagnosis():
    return make_diagnosis(
        issue=IssueType.POSSIBLE_BARREN_PLATEAU,
        confidence=0.9,
        severity="critical",
        evidence=["gradient norm 2.4e-9 for 240 steps"],
        recommendations=["Consider reinitializing parameters"],
    )


@pytest.fixture
def stagnation_diagnosis():
    return make_diagnosis(
        issue=IssueType.STAGNATION,
        confidence=0.7,
        severity="warning",
        evidence=["loss flat for 120 steps"],
        recommendations=["Consider adjusting the learning rate"],
    )


@pytest.fixture
def unstable_diagnosis():
    return make_diagnosis(
        issue=IssueType.UNSTABLE,
        confidence=0.85,
        severity="critical",
        evidence=["loss diverged over 30 steps"],
        recommendations=["Consider reducing the learning rate"],
    )


@pytest.fixture
def noise_dominated_diagnosis():
    return make_diagnosis(
        issue=IssueType.NOISE_DOMINATED,
        confidence=0.75,
        severity="warning",
        evidence=["gradient SNR 0.4 for 50 shot-bearing steps"],
        recommendations=["Consider increasing the shot budget"],
    )


@pytest.fixture
def healthy_diagnosis():
    return make_diagnosis(issue=IssueType.HEALTHY, confidence=0.95, severity="info")


@pytest.fixture
def degraded_diagnosis():
    return make_diagnosis(
        issue=IssueType.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        severity="critical",
        degraded=True,
        degraded_reason="BarrenPlateauDetector raised ZeroDivisionError at step 42",
    )


@pytest.fixture
def bare_context() -> RecoveryContext:
    """A `RecoveryContext` with no optional fields populated."""
    return RecoveryContext(run_id="test-run", step=100)


@pytest.fixture
def full_context() -> RecoveryContext:
    """A `RecoveryContext` with circuit/optimizer/shots/gradient all populated."""
    return RecoveryContext(
        run_id="test-run",
        step=100,
        circuit=CircuitMetadata(n_qubits=8, depth=20, initialization="random_uniform"),
        optimizer=OptimizerMetadata(name="Adam", learning_rate=0.05),
        shots=1000,
        gradient=GradientSnapshot(
            values=None,
            norm_l2=1e-3,
            mean_abs=1e-4,
            variance=1e-6,
            min_value=-2e-4,
            max_value=2e-4,
            median_abs=1e-4,
        ),
        planned_steps=1000,
    )
