"""Unit tests for qml_observer.recovery.strategies.shot_budget."""

from __future__ import annotations

import pytest

from qml_observer.recovery.base import RecoveryContext
from qml_observer.recovery.strategies.shot_budget import (
    _MAX_SHOT_MULTIPLIER,
    ShotBudgetAdjustmentStrategy,
)
from qml_observer.schemas.gradient import GradientSnapshot


def _gradient(mean_abs: float, variance: float) -> GradientSnapshot:
    return GradientSnapshot(
        values=None,
        norm_l2=mean_abs,
        mean_abs=mean_abs,
        variance=variance,
        min_value=-mean_abs,
        max_value=mean_abs,
        median_abs=mean_abs,
    )


class TestConstruction:
    def test_default_target_snr(self):
        assert ShotBudgetAdjustmentStrategy()._target_snr > 0

    def test_custom_target_snr(self):
        assert ShotBudgetAdjustmentStrategy(target_snr=2.0)._target_snr == 2.0

    def test_non_positive_target_snr_raises(self):
        with pytest.raises(ValueError, match="target_snr"):
            ShotBudgetAdjustmentStrategy(target_snr=0.0)
        with pytest.raises(ValueError, match="target_snr"):
            ShotBudgetAdjustmentStrategy(target_snr=-1.0)


class TestAppliesTo:
    def test_applies_to_noise_dominated(self, noise_dominated_diagnosis):
        assert ShotBudgetAdjustmentStrategy().applies_to(noise_dominated_diagnosis) is True

    def test_does_not_apply_to_barren_plateau(self, barren_plateau_diagnosis):
        assert ShotBudgetAdjustmentStrategy().applies_to(barren_plateau_diagnosis) is False

    def test_does_not_apply_to_stagnation(self, stagnation_diagnosis):
        assert ShotBudgetAdjustmentStrategy().applies_to(stagnation_diagnosis) is False

    def test_does_not_apply_to_healthy(self, healthy_diagnosis):
        assert ShotBudgetAdjustmentStrategy().applies_to(healthy_diagnosis) is False


class TestProposeWithFullContext:
    def test_low_snr_recommends_more_shots(self, noise_dominated_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, shots=100, gradient=_gradient(mean_abs=1e-3, variance=1e-3)
        )
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, ctx)
        assert rec is not None
        assert rec.hook_name == "set_shots"
        assert rec.parameters["shots"] > 100

    def test_never_recommends_fewer_shots(self, noise_dominated_diagnosis):
        # A gradient/shot combination whose SNR is already comfortably
        # above target should still never propose a *decrease*.
        ctx = RecoveryContext(
            run_id="r1", step=10, shots=1_000_000, gradient=_gradient(mean_abs=1.0, variance=1e-9)
        )
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["shots"] >= 1_000_000

    def test_multiplier_is_capped(self, noise_dominated_diagnosis):
        # Near-zero SNR should hit the multiplier cap, not blow up.
        ctx = RecoveryContext(
            run_id="r1", step=10, shots=10, gradient=_gradient(mean_abs=1e-12, variance=1e6)
        )
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["shots"] <= 10 * _MAX_SHOT_MULTIPLIER + 1

    def test_zero_snr_hits_cap_without_error(self, noise_dominated_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, shots=50, gradient=_gradient(mean_abs=0.0, variance=1.0)
        )
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["shots"] > 50


class TestProposeWithSparseContext:
    def test_missing_shots_falls_back_to_generic_recommendation(self, noise_dominated_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, gradient=_gradient(1e-3, 1e-3))
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["shots"] > 0

    def test_missing_gradient_falls_back_to_generic_recommendation(self, noise_dominated_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, shots=500)
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["shots"] > 0

    def test_zero_shots_treated_as_missing(self, noise_dominated_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, shots=0, gradient=_gradient(1e-3, 1e-3))
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["shots"] > 0


class TestRecommendationShape:
    def test_priority_in_valid_range(self, noise_dominated_diagnosis, full_context):
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, full_context)
        assert 0.0 <= rec.priority <= 1.0

    def test_rationale_mentions_snr(self, noise_dominated_diagnosis, full_context):
        rec = ShotBudgetAdjustmentStrategy().propose(noise_dominated_diagnosis, full_context)
        assert any("SNR" in r or "snr" in r for r in rec.rationale)
