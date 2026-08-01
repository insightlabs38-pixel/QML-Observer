"""Unit tests for qml_observer.recovery.strategies.natural_gradient."""

from __future__ import annotations

from qml_observer.recovery.base import RecoveryContext
from qml_observer.recovery.strategies.natural_gradient import NaturalGradientStrategy
from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.optimizer import OptimizerMetadata


class TestAppliesTo:
    def test_applies_to_barren_plateau(self, barren_plateau_diagnosis):
        assert NaturalGradientStrategy().applies_to(barren_plateau_diagnosis) is True

    def test_applies_to_stagnation(self, stagnation_diagnosis):
        assert NaturalGradientStrategy().applies_to(stagnation_diagnosis) is True

    def test_does_not_apply_to_unstable(self, unstable_diagnosis):
        assert NaturalGradientStrategy().applies_to(unstable_diagnosis) is False

    def test_does_not_apply_to_noise_dominated(self, noise_dominated_diagnosis):
        assert NaturalGradientStrategy().applies_to(noise_dominated_diagnosis) is False


class TestPropose:
    def test_proposes_natural_gradient_optimizer(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10)
        rec = NaturalGradientStrategy().propose(barren_plateau_diagnosis, ctx)
        assert rec is not None
        assert rec.hook_name == "set_optimizer"
        assert rec.parameters == {"optimizer": "QuantumNaturalGradient"}

    def test_already_natural_gradient_returns_none(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, optimizer=OptimizerMetadata(name="QNSPSA"))
        assert NaturalGradientStrategy().propose(barren_plateau_diagnosis, ctx) is None

    def test_already_natural_gradient_case_insensitive(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(
            run_id="r1", step=10, optimizer=OptimizerMetadata(name="QuantumNaturalGradient")
        )
        assert NaturalGradientStrategy().propose(barren_plateau_diagnosis, ctx) is None

    def test_different_optimizer_still_proposes(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, optimizer=OptimizerMetadata(name="Adam"))
        rec = NaturalGradientStrategy().propose(barren_plateau_diagnosis, ctx)
        assert rec is not None

    def test_priority_capped_below_cheaper_strategies(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        ctx = RecoveryContext(run_id="r1", step=10)
        diag = diagnosis_factory(
            issue=IssueType.POSSIBLE_BARREN_PLATEAU, confidence=1.0, severity="critical"
        )
        rec = NaturalGradientStrategy().propose(diag, ctx)
        assert rec.priority <= 0.4  # deliberately capped, see module docstring

    def test_rationale_mentions_qfim_cost_when_circuit_known(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, circuit=CircuitMetadata(n_qubits=12))
        rec = NaturalGradientStrategy().propose(barren_plateau_diagnosis, ctx)
        assert any("12" in r for r in rec.rationale)

    def test_priority_in_valid_range(self, barren_plateau_diagnosis, stagnation_diagnosis):
        strategy = NaturalGradientStrategy()
        ctx = RecoveryContext(run_id="r1", step=10)
        for diag in (barren_plateau_diagnosis, stagnation_diagnosis):
            rec = strategy.propose(diag, ctx)
            assert 0.0 <= rec.priority <= 1.0
