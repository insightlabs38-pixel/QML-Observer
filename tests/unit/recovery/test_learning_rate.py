"""Unit tests for qml_observer.recovery.strategies.learning_rate."""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext
from qml_observer.recovery.strategies.learning_rate import LearningRateAdjustmentStrategy
from qml_observer.schemas.optimizer import OptimizerMetadata


class TestAppliesTo:
    def test_applies_to_unstable(self, unstable_diagnosis):
        assert LearningRateAdjustmentStrategy().applies_to(unstable_diagnosis) is True

    def test_applies_to_stagnation(self, stagnation_diagnosis):
        assert LearningRateAdjustmentStrategy().applies_to(stagnation_diagnosis) is True

    def test_does_not_apply_to_barren_plateau(self, barren_plateau_diagnosis):
        assert LearningRateAdjustmentStrategy().applies_to(barren_plateau_diagnosis) is False

    def test_does_not_apply_to_noise_dominated(self, noise_dominated_diagnosis):
        assert LearningRateAdjustmentStrategy().applies_to(noise_dominated_diagnosis) is False

    def test_does_not_apply_to_healthy(self, healthy_diagnosis):
        assert LearningRateAdjustmentStrategy().applies_to(healthy_diagnosis) is False


class TestProposeUnstable:
    def test_known_learning_rate_is_halved(self, unstable_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="Adam", learning_rate=0.1)
        )
        rec = LearningRateAdjustmentStrategy().propose(unstable_diagnosis, ctx)
        assert rec is not None
        assert rec.hook_name == "set_learning_rate"
        assert rec.parameters["learning_rate"] == 0.05

    def test_unknown_learning_rate_falls_back_conservatively(self, unstable_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = LearningRateAdjustmentStrategy().propose(unstable_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["learning_rate"] > 0

    def test_zero_learning_rate_treated_as_unknown(self, unstable_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="Adam", learning_rate=0.0)
        )
        rec = LearningRateAdjustmentStrategy().propose(unstable_diagnosis, ctx)
        assert rec.parameters["learning_rate"] > 0

    def test_high_priority_for_confident_instability(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="Adam", learning_rate=0.1)
        )
        diag = diagnosis_factory(issue=IssueType.UNSTABLE, confidence=0.95, severity="critical")
        rec = LearningRateAdjustmentStrategy().propose(diag, ctx)
        assert rec.priority > 0.8


class TestProposeStagnation:
    def test_known_learning_rate_is_doubled(self, stagnation_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="Adam", learning_rate=0.01)
        )
        rec = LearningRateAdjustmentStrategy().propose(stagnation_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["learning_rate"] == 0.02

    def test_unknown_learning_rate_falls_back_to_default(self, stagnation_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = LearningRateAdjustmentStrategy().propose(stagnation_diagnosis, ctx)
        assert rec.parameters["learning_rate"] > 0

    def test_stagnation_priority_lower_than_unstable_at_same_confidence(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="Adam", learning_rate=0.05)
        )
        strategy = LearningRateAdjustmentStrategy()
        unstable = diagnosis_factory(issue=IssueType.UNSTABLE, confidence=0.8, severity="critical")
        stagnant = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.8, severity="warning")
        rec_unstable = strategy.propose(unstable, ctx)
        rec_stagnant = strategy.propose(stagnant, ctx)
        assert rec_stagnant.priority < rec_unstable.priority


class TestRecommendationShape:
    def test_priority_in_valid_range(self, unstable_diagnosis, stagnation_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        strategy = LearningRateAdjustmentStrategy()
        for diag in (unstable_diagnosis, stagnation_diagnosis):
            rec = strategy.propose(diag, ctx)
            assert 0.0 <= rec.priority <= 1.0

    def test_rationale_mentions_current_or_unknown_rate(self, unstable_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = LearningRateAdjustmentStrategy().propose(unstable_diagnosis, ctx)
        assert any("learning rate" in r.lower() for r in rec.rationale)
