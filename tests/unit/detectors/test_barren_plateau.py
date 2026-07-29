"""Unit tests for qml_observer.detectors.barren_plateau.BarrenPlateauDetector."""

import numpy as np
import pytest

from qml_observer.detectors.barren_plateau import BarrenPlateauDetector


class TestConfidenceIntervalEvidence:
    """Milestone 9, Issue #69: gradient norm evidence includes a CI."""

    def test_ci_evidence_present_after_first_gradient(self, run_state, obs_factory):
        d = BarrenPlateauDetector()
        obs = obs_factory(step=0, loss=1.0, gradient=np.array([0.1, -0.2, 0.05]))
        run_state.record(obs)
        d.update(obs, run_state)
        result = d.diagnose()
        assert any("CI on gradient norm" in line for line in result.evidence)

    def test_ci_uses_shot_noise_method_when_shots_present(self, run_state, obs_factory):
        d = BarrenPlateauDetector()
        obs = obs_factory(step=0, loss=1.0, gradient=np.array([0.1, -0.2, 0.05]), shots=50)
        run_state.record(obs)
        d.update(obs, run_state)
        result = d.diagnose()
        assert any("shot-noise-analytic" in line for line in result.evidence)

    def test_ci_uses_parameter_spread_method_without_shots(self, run_state, obs_factory):
        d = BarrenPlateauDetector()
        obs = obs_factory(step=0, loss=1.0, gradient=np.array([0.1, -0.2, 0.05]))
        run_state.record(obs)
        d.update(obs, run_state)
        result = d.diagnose()
        assert any("parameter-spread-analytic" in line for line in result.evidence)

    def test_no_gradient_data_omits_ci_evidence(self, run_state, obs_factory):
        d = BarrenPlateauDetector()
        obs = obs_factory(step=0, loss=1.0)
        run_state.record(obs)
        d.update(obs, run_state)
        result = d.diagnose()
        assert not any("CI on gradient norm" in line for line in result.evidence)

    def test_reset_clears_ci_tracking(self, run_state, obs_factory):
        d = BarrenPlateauDetector()
        obs = obs_factory(step=0, loss=1.0, gradient=np.array([0.1, -0.2, 0.05]))
        run_state.record(obs)
        d.update(obs, run_state)
        d.reset()
        result = d.diagnose()
        assert not any("CI on gradient norm" in line for line in result.evidence)
    def test_defaults(self):
        d = BarrenPlateauDetector()
        assert d.name == "barren_plateau"

    def test_invalid_gradient_threshold_raises(self):
        with pytest.raises(ValueError):
            BarrenPlateauDetector(gradient_threshold=0)
        with pytest.raises(ValueError):
            BarrenPlateauDetector(gradient_threshold=-1)

    def test_invalid_variance_threshold_raises(self):
        with pytest.raises(ValueError):
            BarrenPlateauDetector(variance_threshold=0)

    def test_invalid_loss_improvement_threshold_raises(self):
        with pytest.raises(ValueError):
            BarrenPlateauDetector(loss_improvement_threshold=-1)

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError):
            BarrenPlateauDetector(patience=0)

    def test_default_variance_threshold_derived_from_gradient_threshold(self):
        d = BarrenPlateauDetector(gradient_threshold=1e-4)
        assert d._variance_threshold == pytest.approx(1e-8)


class TestDiagnoseNoData:
    def test_no_data_yields_zero_confidence_not_triggered(self, run_state, obs_factory):
        d = BarrenPlateauDetector()
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence == 0.0


class TestBarrenPlateauScenario:
    def test_sustained_small_gradient_and_stagnant_loss_triggers(self, run_state, obs_factory):
        d = BarrenPlateauDetector(
            gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=10
        )
        rng = np.random.default_rng(0)
        result = None
        for step in range(30):
            obs = obs_factory(
                step=step,
                loss=0.8 + rng.normal(0, 1e-8),
                gradient=rng.normal(0, 1e-6, size=8),
            )
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is True
        assert result.confidence >= 0.6
        assert any("gradient" in e.lower() for e in result.evidence)
        assert result.recommendations

    def test_small_gradient_alone_without_loss_data_never_triggers(self, run_state, obs_factory):
        """A small gradient alone must never be enough (plan.md sec 13)."""
        d = BarrenPlateauDetector(gradient_threshold=1e-3, patience=5)
        rng = np.random.default_rng(1)
        for step in range(20):
            obs = obs_factory(step=step, loss=None, gradient=rng.normal(0, 1e-6, size=8))
            run_state.record(obs)
            d.update(obs, run_state)
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence < 0.6

    def test_healthy_training_does_not_trigger(self, run_state, obs_factory):
        d = BarrenPlateauDetector(
            gradient_threshold=1e-3, loss_improvement_threshold=1e-4, patience=10
        )
        rng = np.random.default_rng(2)
        loss = 1.0
        result = None
        for step in range(30):
            loss *= 0.9
            obs = obs_factory(step=step, loss=loss, gradient=rng.normal(0, 0.1, size=8))
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is False

    def test_reset_clears_state(self, run_state, obs_factory):
        d = BarrenPlateauDetector(patience=5)
        rng = np.random.default_rng(3)
        for step in range(10):
            obs = obs_factory(step=step, loss=0.5, gradient=rng.normal(0, 1e-6, size=4))
            run_state.record(obs)
            d.update(obs, run_state)
        d.reset()
        result = d.diagnose()
        assert result.confidence == 0.0
        assert result.triggered is False
