"""Unit tests for qml_observer.recovery.strategies.optimizer_switching."""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext
from qml_observer.recovery.strategies.optimizer_switching import OptimizerSwitchingStrategy
from qml_observer.schemas.optimizer import OptimizerMetadata


class TestAppliesTo:
    def test_applies_to_unstable(self, unstable_diagnosis):
        assert OptimizerSwitchingStrategy().applies_to(unstable_diagnosis) is True

    def test_applies_to_stagnation(self, stagnation_diagnosis):
        assert OptimizerSwitchingStrategy().applies_to(stagnation_diagnosis) is True

    def test_does_not_apply_to_barren_plateau(self, barren_plateau_diagnosis):
        assert OptimizerSwitchingStrategy().applies_to(barren_plateau_diagnosis) is False

    def test_does_not_apply_to_noise_dominated(self, noise_dominated_diagnosis):
        assert OptimizerSwitchingStrategy().applies_to(noise_dominated_diagnosis) is False


class TestProposeForInstability:
    def test_adaptive_optimizer_switches_to_conservative(self, unstable_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="Adam", learning_rate=0.1)
        )
        rec = OptimizerSwitchingStrategy().propose(unstable_diagnosis, ctx)
        assert rec is not None
        assert rec.hook_name == "set_optimizer"
        assert rec.parameters["optimizer"] == "GradientDescent"

    def test_conservative_optimizer_switches_to_perturbation(self, unstable_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="GradientDescent")
        )
        rec = OptimizerSwitchingStrategy().propose(unstable_diagnosis, ctx)
        assert rec.parameters["optimizer"] == "SPSA"

    def test_perturbation_optimizer_switches_to_conservative(self, unstable_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, optimizer=OptimizerMetadata(name="SPSA"))
        rec = OptimizerSwitchingStrategy().propose(unstable_diagnosis, ctx)
        assert rec.parameters["optimizer"] == "GradientDescent"

    def test_unknown_optimizer_switches_to_perturbation(self, unstable_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="SomeCustomOptimizer")
        )
        rec = OptimizerSwitchingStrategy().propose(unstable_diagnosis, ctx)
        assert rec.parameters["optimizer"] == "SPSA"

    def test_missing_optimizer_still_proposes(self, unstable_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = OptimizerSwitchingStrategy().propose(unstable_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters["optimizer"] == "SPSA"

    def test_case_insensitive_matching(self, unstable_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, optimizer=OptimizerMetadata(name="adam"))
        rec = OptimizerSwitchingStrategy().propose(unstable_diagnosis, ctx)
        assert rec.parameters["optimizer"] == "GradientDescent"


class TestProposeForStagnation:
    def test_conservative_optimizer_switches_to_adaptive(self, stagnation_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="GradientDescent")
        )
        rec = OptimizerSwitchingStrategy().propose(stagnation_diagnosis, ctx)
        assert rec.parameters["optimizer"] == "Adam"

    def test_adaptive_optimizer_switches_to_perturbation(self, stagnation_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, optimizer=OptimizerMetadata(name="Adam"))
        rec = OptimizerSwitchingStrategy().propose(stagnation_diagnosis, ctx)
        assert rec.parameters["optimizer"] == "SPSA"

    def test_perturbation_optimizer_switches_to_adaptive(self, stagnation_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, optimizer=OptimizerMetadata(name="SPSA"))
        rec = OptimizerSwitchingStrategy().propose(stagnation_diagnosis, ctx)
        assert rec.parameters["optimizer"] == "Adam"

    def test_missing_optimizer_defaults_to_adaptive(self, stagnation_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = OptimizerSwitchingStrategy().propose(stagnation_diagnosis, ctx)
        assert rec.parameters["optimizer"] == "Adam"

    def test_stagnation_priority_lower_than_instability(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        strategy = OptimizerSwitchingStrategy()
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="GradientDescent")
        )
        unstable = diagnosis_factory(issue=IssueType.UNSTABLE, confidence=0.8, severity="critical")
        stagnant = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.8, severity="warning")
        assert strategy.propose(stagnant, ctx).priority < strategy.propose(unstable, ctx).priority


class TestRecommendationShape:
    def test_priority_in_valid_range(self, unstable_diagnosis, stagnation_diagnosis):
        strategy = OptimizerSwitchingStrategy()
        ctx = RecoveryContext(run_id="r1", step=10)
        for diag in (unstable_diagnosis, stagnation_diagnosis):
            rec = strategy.propose(diag, ctx)
            assert 0.0 <= rec.priority <= 1.0

    def test_rationale_non_empty(self, unstable_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = OptimizerSwitchingStrategy().propose(unstable_diagnosis, ctx)
        assert len(rec.rationale) > 0
