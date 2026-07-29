"""Unit tests for qml_observer.detectors.noise.NoiseDetector."""

import numpy as np
import pytest

from qml_observer.detectors.noise import NoiseDetector


class TestConstruction:
    def test_defaults(self):
        d = NoiseDetector()
        assert d.name == "noise"

    def test_invalid_snr_threshold_raises(self):
        with pytest.raises(ValueError):
            NoiseDetector(snr_threshold=0)
        with pytest.raises(ValueError):
            NoiseDetector(snr_threshold=-1)

    def test_invalid_patience_raises(self):
        with pytest.raises(ValueError):
            NoiseDetector(patience=0)


class TestDiagnoseNoData:
    def test_no_data_yields_zero_confidence_not_triggered(self):
        d = NoiseDetector()
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence == 0.0


class TestAbstainsWithoutShots:
    def test_analytic_execution_never_triggers(self, run_state, obs_factory):
        """shots=None (analytic) carries no shot-noise info; must abstain."""
        d = NoiseDetector(patience=5)
        rng = np.random.default_rng(0)
        for step in range(50):
            obs = obs_factory(step=step, loss=0.5, gradient=rng.normal(0, 1e-7, size=8))
            run_state.record(obs)
            d.update(obs, run_state)
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence == 0.0

    def test_zero_shots_is_treated_like_no_shots(self, run_state, obs_factory):
        d = NoiseDetector(patience=3)
        rng = np.random.default_rng(0)
        for step in range(10):
            obs = obs_factory(step=step, loss=0.5, gradient=rng.normal(0, 0.5, size=8), shots=0)
            run_state.record(obs)
            d.update(obs, run_state)
        result = d.diagnose()
        assert result.triggered is False


class TestLowShotBudgetTriggers:
    def test_sustained_low_snr_with_few_shots_triggers(self, run_state, obs_factory):
        """A moderate gradient estimated from very few shots is unreliable."""
        d = NoiseDetector(snr_threshold=1.0, patience=10)
        rng = np.random.default_rng(1)
        result = None
        for step in range(30):
            # Large per-parameter spread (variance) relative to the norm,
            # estimated from a tiny shot budget -> large uncertainty ->
            # low SNR.
            gradient = rng.normal(0, 0.5, size=200)
            obs = obs_factory(step=step, loss=0.5, gradient=gradient, shots=1)
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is True
        assert result.confidence > 0.5

    def test_many_shots_does_not_trigger(self, run_state, obs_factory):
        """The same gradient scale, estimated from a large shot budget, is trustworthy."""
        d = NoiseDetector(snr_threshold=1.0, patience=10)
        rng = np.random.default_rng(1)
        result = None
        for step in range(30):
            gradient = rng.normal(0, 0.5, size=200)
            obs = obs_factory(step=step, loss=0.5, gradient=gradient, shots=100_000)
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is False


class TestDoesNotConflateWithCollapse:
    def test_genuinely_collapsed_gradient_does_not_trigger_noise(self, run_state, obs_factory):
        """A real barren-plateau-scale collapse should not be flagged as noise,
        even with a small shot budget: both the gradient norm and its
        variance collapse together, so the shot-noise floor collapses too.
        """
        d = NoiseDetector(snr_threshold=1.0, patience=10)
        rng = np.random.default_rng(2)
        result = None
        for step in range(30):
            gradient = rng.normal(0, 1e-6, size=200)
            obs = obs_factory(step=step, loss=0.8, gradient=gradient, shots=20)
            run_state.record(obs)
            d.update(obs, run_state)
            result = d.diagnose()
        assert result.triggered is False


class TestReset:
    def test_reset_clears_state(self, run_state, obs_factory):
        d = NoiseDetector(patience=3)
        rng = np.random.default_rng(0)
        for step in range(5):
            gradient = rng.normal(0, 0.5, size=8)
            obs = obs_factory(step=step, loss=0.5, gradient=gradient, shots=2)
            run_state.record(obs)
            d.update(obs, run_state)
        d.reset()
        result = d.diagnose()
        assert result.triggered is False
        assert result.confidence == 0.0
