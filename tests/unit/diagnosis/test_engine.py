"""Unit tests for qml_observer.diagnosis.engine.DiagnosisEngine."""

import numpy as np
import pytest

from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.convergence import ConvergenceDetector
from qml_observer.detectors.stagnation import StagnationDetector
from qml_observer.diagnosis.engine import DiagnosisEngine
from qml_observer.schemas.diagnosis import IssueType


class TestConstruction:
    def test_empty_detector_list_allowed(self):
        engine = DiagnosisEngine([])
        assert engine.detectors == []

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            DiagnosisEngine("not-a-list")

    def test_non_detector_element_raises(self):
        with pytest.raises(TypeError):
            DiagnosisEngine([object()])


class TestEvaluateNoDetectors:
    def test_insufficient_evidence_with_no_detectors(self, run_state, obs_factory):
        engine = DiagnosisEngine([])
        obs = obs_factory(step=0, loss=1.0)
        run_state.record(obs)
        result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.INSUFFICIENT_EVIDENCE
        assert result.confidence == 0.0


class TestEvaluateNoData:
    def test_insufficient_evidence_before_any_step(self, run_state, obs_factory):
        engine = DiagnosisEngine([BarrenPlateauDetector()])
        obs = obs_factory(step=0, loss=None)
        run_state.record(obs)
        result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.INSUFFICIENT_EVIDENCE


class TestEvaluateBarrenPlateau:
    def test_reports_possible_barren_plateau_with_critical_severity(self, run_state, obs_factory):
        engine = DiagnosisEngine(
            [
                BarrenPlateauDetector(
                    gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=10
                ),
                StagnationDetector(patience=10),
                ConvergenceDetector(loss_threshold=0.05, gradient_threshold=1e-3, patience=10),
            ]
        )
        rng = np.random.default_rng(0)
        result = None
        for step in range(30):
            obs = obs_factory(
                step=step, loss=0.8 + rng.normal(0, 1e-8), gradient=rng.normal(0, 1e-6, size=8)
            )
            run_state.record(obs)
            result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.POSSIBLE_BARREN_PLATEAU
        assert result.severity == "critical"
        assert any("[barren_plateau]" in e for e in result.evidence)
        assert result.recommendations


class TestEvaluateConvergence:
    def test_reports_converged_not_barren_plateau_at_low_loss(self, run_state, obs_factory):
        engine = DiagnosisEngine(
            [
                BarrenPlateauDetector(
                    gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=10
                ),
                ConvergenceDetector(loss_threshold=0.05, gradient_threshold=1e-3, patience=10),
            ]
        )
        rng = np.random.default_rng(1)
        result = None
        for step in range(30):
            obs = obs_factory(step=step, loss=0.001, gradient=rng.normal(0, 1e-6, size=8))
            run_state.record(obs)
            result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.CONVERGED
        assert result.severity == "info"


class TestEvaluateHealthy:
    def test_reports_healthy_when_nothing_triggers(self, run_state, obs_factory):
        engine = DiagnosisEngine(
            [
                BarrenPlateauDetector(
                    gradient_threshold=1e-6, loss_improvement_threshold=1e-8, patience=10
                ),
                StagnationDetector(loss_threshold=1e-8, patience=10),
                ConvergenceDetector(loss_threshold=1e-8, gradient_threshold=1e-6, patience=10),
            ]
        )
        loss = 1.0
        rng = np.random.default_rng(2)
        result = None
        for step in range(20):
            loss *= 0.9
            obs = obs_factory(step=step, loss=loss, gradient=rng.normal(0, 0.2, size=8))
            run_state.record(obs)
            result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.HEALTHY
        assert result.severity == "info"


class TestEvaluateInstability:
    """Addendum §7 / Milestone 7 beta review: `_check_instability` must
    override the headline diagnosis whenever loss or gradient norm is
    NaN/Inf, ahead of (and regardless of) whatever the detectors
    themselves report."""

    def test_nan_loss_reports_unstable(self, run_state, obs_factory):
        engine = DiagnosisEngine([BarrenPlateauDetector(patience=3)])
        obs = obs_factory(step=0, loss=float("nan"), gradient=np.array([0.1, 0.2]))
        run_state.record(obs)
        result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.UNSTABLE
        assert result.severity == "critical"
        assert result.confidence == 1.0
        assert result.recommendations

    def test_inf_loss_reports_unstable(self, run_state, obs_factory):
        engine = DiagnosisEngine([BarrenPlateauDetector(patience=3)])
        obs = obs_factory(step=0, loss=float("inf"), gradient=np.array([0.1, 0.2]))
        run_state.record(obs)
        result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.UNSTABLE

    def test_non_finite_gradient_norm_reports_unstable(self, run_state, obs_factory):
        engine = DiagnosisEngine([BarrenPlateauDetector(patience=3)])
        obs = obs_factory(step=0, loss=0.5, gradient=np.array([float("inf"), 1.0]))
        run_state.record(obs)
        result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.UNSTABLE

    def test_unstable_overrides_converged(self, run_state, obs_factory):
        """A NaN loss must never be reported as CONVERGED even if a prior
        window of genuinely-converged steps preceded it."""
        engine = DiagnosisEngine(
            [ConvergenceDetector(loss_threshold=0.05, gradient_threshold=1e-3, patience=5)]
        )
        result = None
        for step in range(10):
            obs = obs_factory(step=step, loss=0.001, gradient=np.full(4, 1e-6))
            run_state.record(obs)
            result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.CONVERGED  # sanity check on the setup

        obs = obs_factory(step=10, loss=float("nan"), gradient=np.full(4, 1e-6))
        run_state.record(obs)
        result = engine.evaluate(obs, run_state)
        assert result.issue == IssueType.UNSTABLE

    def test_finite_values_are_not_reported_as_unstable(self, run_state, obs_factory):
        engine = DiagnosisEngine([BarrenPlateauDetector(patience=3)])
        obs = obs_factory(step=0, loss=0.5, gradient=np.array([0.1, 0.2]))
        run_state.record(obs)
        result = engine.evaluate(obs, run_state)
        assert result.issue != IssueType.UNSTABLE

    def test_detectors_still_update_on_an_unstable_step(self, run_state, obs_factory):
        """Detector rolling state must keep advancing even on an unstable
        step, so the run can be correctly diagnosed if it later recovers
        to finite values."""
        detector = BarrenPlateauDetector(patience=3)
        engine = DiagnosisEngine([detector])
        obs = obs_factory(step=0, loss=float("nan"), gradient=np.array([0.1, 0.2]))
        run_state.record(obs)
        engine.evaluate(obs, run_state)
        # The detector's own internal gradient-norm window recorded a step.
        assert detector.diagnose().evidence  # non-empty: update() ran


class TestReset:
    def test_reset_clears_all_driven_detectors(self, run_state, obs_factory):
        detector = BarrenPlateauDetector(patience=3)
        engine = DiagnosisEngine([detector])
        rng = np.random.default_rng(3)
        for step in range(10):
            obs = obs_factory(step=step, loss=0.5, gradient=rng.normal(0, 1e-6, size=4))
            run_state.record(obs)
            engine.evaluate(obs, run_state)
        engine.reset()
        assert detector.diagnose().confidence == 0.0
