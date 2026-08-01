"""Unit tests for qml_observer.recovery.strategies.reinitialization."""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext
from qml_observer.recovery.strategies.reinitialization import ParameterReinitializationStrategy
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import IssueType


class TestAppliesTo:
    def test_applies_to_barren_plateau(self, barren_plateau_diagnosis):
        assert ParameterReinitializationStrategy().applies_to(barren_plateau_diagnosis) is True

    def test_applies_to_stagnation(self, stagnation_diagnosis):
        assert ParameterReinitializationStrategy().applies_to(stagnation_diagnosis) is True

    def test_does_not_apply_to_noise_dominated(self, noise_dominated_diagnosis):
        assert ParameterReinitializationStrategy().applies_to(noise_dominated_diagnosis) is False

    def test_does_not_apply_to_healthy(self, healthy_diagnosis):
        assert ParameterReinitializationStrategy().applies_to(healthy_diagnosis) is False

    def test_does_not_apply_to_unstable(self, unstable_diagnosis):
        assert ParameterReinitializationStrategy().applies_to(unstable_diagnosis) is False


class TestProposeBarrenPlateau:
    def test_generic_initialization_recommends_reduced_domain(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, circuit=CircuitMetadata(initialization="random_uniform")
        )
        rec = ParameterReinitializationStrategy().propose(barren_plateau_diagnosis, ctx)
        assert rec is not None
        assert rec.hook_name == "reinitialize_parameters"
        assert rec.parameters == {"initialization": "reduced_domain"}
        assert rec.priority > 0.5

    def test_unknown_initialization_treated_as_generic(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, circuit=None)
        rec = ParameterReinitializationStrategy().propose(barren_plateau_diagnosis, ctx)
        assert rec is not None
        assert rec.parameters == {"initialization": "reduced_domain"}

    def test_already_barren_plateau_aware_initialization_lower_priority(
        self, barren_plateau_diagnosis
    ):
        ctx_generic = RecoveryContext(
            run_id="r1", step=10, circuit=CircuitMetadata(initialization="random_uniform")
        )
        ctx_aware = RecoveryContext(
            run_id="r1", step=10, circuit=CircuitMetadata(initialization="reduced_domain")
        )
        strategy = ParameterReinitializationStrategy()
        rec_generic = strategy.propose(barren_plateau_diagnosis, ctx_generic)
        rec_aware = strategy.propose(barren_plateau_diagnosis, ctx_aware)
        assert rec_aware.priority < rec_generic.priority
        assert rec_aware.parameters == {}  # no specific initialization proposed


class TestProposeStagnation:
    def test_stagnation_recommends_reinit_without_initialization_change(self, stagnation_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = ParameterReinitializationStrategy().propose(stagnation_diagnosis, ctx)
        assert rec is not None
        assert rec.hook_name == "reinitialize_parameters"
        assert rec.parameters == {}
        assert 0.0 < rec.priority <= 0.6


class TestRecommendationShape:
    def test_priority_stays_in_valid_range(self, barren_plateau_diagnosis, diagnosis_factory):
        ctx = RecoveryContext(run_id="r1", step=10)
        strategy = ParameterReinitializationStrategy()
        for confidence in (0.0, 0.5, 1.0):
            diag = diagnosis_factory(
                issue=IssueType.POSSIBLE_BARREN_PLATEAU, confidence=confidence, severity="critical"
            )
            rec = strategy.propose(diag, ctx)
            assert 0.0 <= rec.priority <= 1.0

    def test_rationale_is_non_empty(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = ParameterReinitializationStrategy().propose(barren_plateau_diagnosis, ctx)
        assert len(rec.rationale) > 0
