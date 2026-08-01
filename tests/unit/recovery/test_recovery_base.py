"""Unit tests for qml_observer.recovery.base."""

from __future__ import annotations

import pytest

from qml_observer.recovery.base import (
    RecoveryContext,
    RecoveryOutcome,
    RecoveryRecommendation,
    RecoveryStrategy,
)
from qml_observer.schemas.diagnosis import DiagnosisResult


class TestRecoveryContextValidation:
    def test_minimal_valid_construction(self):
        ctx = RecoveryContext(run_id="r1", step=0)
        assert ctx.run_id == "r1"
        assert ctx.step == 0
        assert ctx.circuit is None
        assert ctx.optimizer is None
        assert ctx.shots is None
        assert ctx.gradient is None
        assert ctx.planned_steps is None

    def test_empty_run_id_raises(self):
        with pytest.raises(ValueError, match="run_id"):
            RecoveryContext(run_id="", step=0)

    def test_negative_step_raises(self):
        with pytest.raises(ValueError, match="step"):
            RecoveryContext(run_id="r1", step=-1)

    def test_wrong_circuit_type_raises(self):
        with pytest.raises(TypeError, match="circuit"):
            RecoveryContext(run_id="r1", step=0, circuit="not-a-circuit")  # type: ignore[arg-type]

    def test_wrong_optimizer_type_raises(self):
        with pytest.raises(TypeError, match="optimizer"):
            RecoveryContext(run_id="r1", step=0, optimizer="not-an-optimizer")  # type: ignore[arg-type]

    def test_negative_shots_raises(self):
        with pytest.raises(ValueError, match="shots"):
            RecoveryContext(run_id="r1", step=0, shots=-5)

    def test_wrong_gradient_type_raises(self):
        with pytest.raises(TypeError, match="gradient"):
            RecoveryContext(run_id="r1", step=0, gradient="not-a-gradient")  # type: ignore[arg-type]

    def test_negative_planned_steps_raises(self):
        with pytest.raises(ValueError, match="planned_steps"):
            RecoveryContext(run_id="r1", step=0, planned_steps=-1)

    def test_full_context_construction(self, full_context):
        assert full_context.circuit is not None
        assert full_context.optimizer is not None
        assert full_context.shots == 1000
        assert full_context.gradient is not None
        assert full_context.planned_steps == 1000


class TestRecoveryRecommendationValidation:
    def test_minimal_valid_construction(self):
        rec = RecoveryRecommendation(
            strategy_name="test_strategy", description="do the thing", priority=0.5
        )
        assert rec.parameters == {}
        assert rec.rationale == []
        assert rec.hook_name is None

    def test_empty_strategy_name_raises(self):
        with pytest.raises(ValueError, match="strategy_name"):
            RecoveryRecommendation(strategy_name="", description="x", priority=0.5)

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            RecoveryRecommendation(strategy_name="s", description="", priority=0.5)

    def test_priority_out_of_range_raises(self):
        with pytest.raises(ValueError, match="priority"):
            RecoveryRecommendation(strategy_name="s", description="x", priority=1.5)

    def test_negative_priority_raises(self):
        with pytest.raises(ValueError, match="priority"):
            RecoveryRecommendation(strategy_name="s", description="x", priority=-0.1)

    def test_non_dict_parameters_raises(self):
        with pytest.raises(TypeError, match="parameters"):
            RecoveryRecommendation(
                strategy_name="s", description="x", priority=0.5, parameters=["not", "a", "dict"]
            )

    def test_non_str_rationale_item_raises(self):
        with pytest.raises(TypeError, match="rationale"):
            RecoveryRecommendation(
                strategy_name="s", description="x", priority=0.5, rationale=[1, 2]
            )

    def test_empty_hook_name_raises(self):
        with pytest.raises(ValueError, match="hook_name"):
            RecoveryRecommendation(strategy_name="s", description="x", priority=0.5, hook_name="")

    def test_full_construction(self):
        rec = RecoveryRecommendation(
            strategy_name="s",
            description="x",
            priority=0.8,
            parameters={"learning_rate": 0.01},
            rationale=["because reasons"],
            hook_name="set_learning_rate",
        )
        assert rec.parameters == {"learning_rate": 0.01}
        assert rec.hook_name == "set_learning_rate"


class TestRecoveryOutcomeValidation:
    def test_minimal_valid_construction(self):
        outcome = RecoveryOutcome(strategy_name="s", applied=True, message="done")
        assert outcome.applied is True

    def test_empty_strategy_name_raises(self):
        with pytest.raises(ValueError, match="strategy_name"):
            RecoveryOutcome(strategy_name="", applied=True, message="done")

    def test_non_bool_applied_raises(self):
        with pytest.raises(TypeError, match="applied"):
            RecoveryOutcome(strategy_name="s", applied="yes", message="done")  # type: ignore[arg-type]

    def test_non_str_message_raises(self):
        with pytest.raises(TypeError, match="message"):
            RecoveryOutcome(strategy_name="s", applied=True, message=123)  # type: ignore[arg-type]


class TestRecoveryStrategyIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            RecoveryStrategy()  # type: ignore[abstract]

    def test_subclass_must_implement_both_methods(self):
        class Incomplete(RecoveryStrategy):
            name = "incomplete"

            def applies_to(self, diagnosis: DiagnosisResult) -> bool:
                return True

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self, healthy_diagnosis, bare_context):
        class AlwaysProposes(RecoveryStrategy):
            name = "always-proposes"

            def applies_to(self, diagnosis: DiagnosisResult) -> bool:
                return True

            def propose(self, diagnosis, context):
                return RecoveryRecommendation(
                    strategy_name=self.name, description="proposed", priority=0.5
                )

        strategy = AlwaysProposes()
        assert strategy.applies_to(healthy_diagnosis) is True
        recommendation = strategy.propose(healthy_diagnosis, bare_context)
        assert recommendation is not None
        assert recommendation.strategy_name == "always-proposes"
