"""Unit tests for qml_observer.recovery.strategies.depth_reduction."""

from __future__ import annotations

import pytest

from qml_observer.recovery.base import RecoveryContext
from qml_observer.recovery.strategies.depth_reduction import AnsatzDepthReductionStrategy
from qml_observer.schemas.circuit import CircuitMetadata


class TestConstruction:
    def test_default_reduction_fraction(self):
        assert AnsatzDepthReductionStrategy()._reduction_fraction == 0.5

    def test_custom_reduction_fraction(self):
        assert AnsatzDepthReductionStrategy(reduction_fraction=0.25)._reduction_fraction == 0.25

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_out_of_range_reduction_fraction_raises(self, bad):
        with pytest.raises(ValueError, match="reduction_fraction"):
            AnsatzDepthReductionStrategy(reduction_fraction=bad)


class TestAppliesTo:
    def test_applies_to_barren_plateau(self, barren_plateau_diagnosis):
        assert AnsatzDepthReductionStrategy().applies_to(barren_plateau_diagnosis) is True

    def test_does_not_apply_to_stagnation(self, stagnation_diagnosis):
        assert AnsatzDepthReductionStrategy().applies_to(stagnation_diagnosis) is False

    def test_does_not_apply_to_unstable(self, unstable_diagnosis):
        assert AnsatzDepthReductionStrategy().applies_to(unstable_diagnosis) is False

    def test_does_not_apply_to_noise_dominated(self, noise_dominated_diagnosis):
        assert AnsatzDepthReductionStrategy().applies_to(noise_dominated_diagnosis) is False


class TestPropose:
    def test_reduces_depth_by_default_fraction(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, circuit=CircuitMetadata(depth=20))
        rec = AnsatzDepthReductionStrategy().propose(barren_plateau_diagnosis, ctx)
        assert rec is not None
        assert rec.hook_name == "set_circuit_depth"
        assert rec.parameters == {"depth": 10}

    def test_reduces_depth_by_custom_fraction(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, circuit=CircuitMetadata(depth=20))
        rec = AnsatzDepthReductionStrategy(reduction_fraction=0.25).propose(
            barren_plateau_diagnosis, ctx
        )
        assert rec.parameters == {"depth": 15}

    def test_never_goes_below_min_depth(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, circuit=CircuitMetadata(depth=1))
        rec = AnsatzDepthReductionStrategy().propose(barren_plateau_diagnosis, ctx)
        assert rec is None  # already at the floor, nothing to propose

    def test_missing_circuit_returns_none(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, circuit=None)
        assert AnsatzDepthReductionStrategy().propose(barren_plateau_diagnosis, ctx) is None

    def test_missing_depth_returns_none(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, circuit=CircuitMetadata(n_qubits=4))
        assert AnsatzDepthReductionStrategy().propose(barren_plateau_diagnosis, ctx) is None

    def test_deep_circuit_gets_priority_boost(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        strategy = AnsatzDepthReductionStrategy()
        diag = diagnosis_factory(
            issue=IssueType.POSSIBLE_BARREN_PLATEAU, confidence=0.7, severity="critical"
        )
        shallow = strategy.propose(
            diag, RecoveryContext(run_id="r1", step=10, circuit=CircuitMetadata(depth=5))
        )
        deep = strategy.propose(
            diag, RecoveryContext(run_id="r1", step=10, circuit=CircuitMetadata(depth=50))
        )
        assert deep.priority > shallow.priority

    def test_priority_in_valid_range(self, barren_plateau_diagnosis):
        ctx = RecoveryContext(run_id="r1", step=10, circuit=CircuitMetadata(depth=100))
        rec = AnsatzDepthReductionStrategy().propose(barren_plateau_diagnosis, ctx)
        assert 0.0 <= rec.priority <= 1.0
