"""Unit tests for qml_observer.detectors.stagnation.StagnationDetector."""

import numpy as np
import pytest

from qml_observer.detectors.stagnation import StagnationDetector


class TestConstruction:
    def test_defaults(self):
        d = StagnationDetector()
        assert d.name == "stagnation"

    def test_invalid_loss_threshold_raises(self):
        with pytest.raises(ValueError):
            StagnationDetector(loss_threshold=-1)

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError):
            StagnationDetector(patience=0)


class TestDiagnoseNoData:
    def test_no_data_yields_zero_confidence_not_triggered(self):
        d = StagnationDetector()
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence == 0.0


class TestFrozenLearningRate:
    def test_zero_learning_rate_triggers_immediately_with_full_confidence(
        self, run_state, obs_factory
    ):
        d = StagnationDetector(patience=50)
        obs = obs_factory(step=0, loss=0.5, learning_rate=0.0)
        run_state.record(obs)
        d.update(obs, run_state)
        result = d.diagnose()
        assert result.triggered is True
        assert result.confidence == 1.0
        assert result.recommendations

    def test_nonzero_learning_rate_alone_does_not_trigger(self, run_state, obs_factory):
        d = StagnationDetector(patience=50)
        obs = obs_factory(step=0, loss=0.5, learning_rate=0.01)
        run_state.record(obs)
        d.update(obs, run_state)
        result = d.diagnose()
        assert result.triggered is False


class TestFrozenLossAndParameters:
    def test_stagnant_loss_and_frozen_parameters_trigger(self, run_state, obs_factory):
        d = StagnationDetector(loss_threshold=1e-6, patience=10)
        frozen_params = np.ones(5)
        result = None
        for step in range(20):
            obs = obs_factory(step=step, loss=0.5, parameters=frozen_params.copy())
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is True

    def test_moving_parameters_do_not_trigger(self, run_state, obs_factory):
        d = StagnationDetector(loss_threshold=1e-6, patience=10)
        rng = np.random.default_rng(0)
        result = None
        for step in range(20):
            params = rng.normal(size=5)
            obs = obs_factory(step=step, loss=0.5, parameters=params)
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is False

    def test_healthy_decreasing_loss_does_not_trigger(self, run_state, obs_factory):
        d = StagnationDetector(loss_threshold=1e-6, patience=10)
        loss = 1.0
        result = None
        for step in range(20):
            loss *= 0.9
            obs = obs_factory(step=step, loss=loss, parameters=np.full(3, loss))
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is False


class TestReset:
    def test_reset_clears_state(self, run_state, obs_factory):
        d = StagnationDetector(patience=5)
        obs = obs_factory(step=0, loss=0.5, learning_rate=0.0)
        run_state.record(obs)
        d.update(obs, run_state)
        d.reset()
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence == 0.0
