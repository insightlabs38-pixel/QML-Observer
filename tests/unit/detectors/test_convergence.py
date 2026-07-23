"""Unit tests for qml_observer.detectors.convergence.ConvergenceDetector."""

import numpy as np
import pytest

from qml_observer.detectors.convergence import ConvergenceDetector


class TestConstruction:
    def test_defaults(self):
        d = ConvergenceDetector()
        assert d.name == "convergence"

    def test_invalid_loss_threshold_raises(self):
        with pytest.raises(ValueError):
            ConvergenceDetector(loss_threshold=-1)

    def test_invalid_gradient_threshold_raises(self):
        with pytest.raises(ValueError):
            ConvergenceDetector(gradient_threshold=0)

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError):
            ConvergenceDetector(patience=0)


class TestDiagnoseNoData:
    def test_no_data_yields_zero_confidence_not_triggered(self):
        d = ConvergenceDetector()
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence == 0.0


class TestConvergenceScenario:
    def test_low_loss_and_small_gradient_persisted_triggers(self, run_state, obs_factory):
        d = ConvergenceDetector(loss_threshold=0.05, gradient_threshold=1e-3, patience=10)
        rng = np.random.default_rng(0)
        result = None
        for step in range(20):
            obs = obs_factory(step=step, loss=0.001, gradient=rng.normal(0, 1e-5, size=6))
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is True
        assert result.confidence >= 0.7
        assert result.recommendations

    def test_does_not_confuse_gradient_collapse_at_high_loss_with_convergence(
        self, run_state, obs_factory
    ):
        """Small gradient + poor (high) loss must not read as convergence."""
        d = ConvergenceDetector(loss_threshold=0.05, gradient_threshold=1e-3, patience=10)
        rng = np.random.default_rng(1)
        result = None
        for step in range(20):
            obs = obs_factory(step=step, loss=0.9, gradient=rng.normal(0, 1e-6, size=6))
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is False

    def test_insufficient_persistence_does_not_trigger(self, run_state, obs_factory):
        d = ConvergenceDetector(loss_threshold=0.05, gradient_threshold=1e-3, patience=50)
        obs = obs_factory(step=0, loss=0.001, gradient=np.zeros(4))
        run_state.record(obs)
        d.update(obs, run_state)
        result = d.diagnose()
        assert result.triggered is False
        assert 0.0 < result.confidence < 1.0


class TestReset:
    def test_reset_clears_state(self, run_state, obs_factory):
        d = ConvergenceDetector(patience=5)
        obs = obs_factory(step=0, loss=0.0001, gradient=np.zeros(3))
        run_state.record(obs)
        d.update(obs, run_state)
        d.reset()
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence == 0.0
