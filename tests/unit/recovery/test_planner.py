"""Unit tests for qml_observer.recovery.planner.RecoveryPlanner."""

from __future__ import annotations

import logging

import pytest

from qml_observer.recovery.base import RecoveryRecommendation, RecoveryStrategy
from qml_observer.recovery.planner import RecoveryPlanner
from qml_observer.schemas.diagnosis import DiagnosisResult


class _AlwaysApplies(RecoveryStrategy):
    def __init__(self, name: str, priority: float, applies: bool = True):
        self.name = name
        self._priority = priority
        self._applies = applies

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return self._applies

    def propose(self, diagnosis, context):
        return RecoveryRecommendation(
            strategy_name=self.name, description=f"do {self.name}", priority=self._priority
        )


class _ReturnsNone(RecoveryStrategy):
    name = "returns_none"

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return True

    def propose(self, diagnosis, context):
        return None


class _RaisesOnApplicable(RecoveryStrategy):
    name = "raises_applicable"

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        raise RuntimeError("boom in applies_to")

    def propose(self, diagnosis, context):
        raise AssertionError("should never be called")


class _RaisesOnPropose(RecoveryStrategy):
    name = "raises_propose"

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return True

    def propose(self, diagnosis, context):
        raise RuntimeError("boom in propose")


class TestConstruction:
    def test_empty_planner_returns_no_recommendations(self, healthy_diagnosis, bare_context):
        planner = RecoveryPlanner()
        assert planner.recommend(healthy_diagnosis, bare_context) == []

    def test_none_strategies_is_same_as_empty(self, healthy_diagnosis, bare_context):
        planner = RecoveryPlanner(None)
        assert planner.recommend(healthy_diagnosis, bare_context) == []

    def test_rejects_non_strategy_elements(self):
        with pytest.raises(TypeError, match="RecoveryStrategy"):
            RecoveryPlanner(["not a strategy"])  # type: ignore[list-item]

    def test_strategies_property_returns_copy(self):
        strategy = _AlwaysApplies("s1", 0.5)
        planner = RecoveryPlanner([strategy])
        strategies = planner.strategies
        strategies.append(_AlwaysApplies("s2", 0.1))
        assert len(planner.strategies) == 1  # mutation of the returned list is not reflected


class TestRecommend:
    def test_applicable_strategy_is_included(self, healthy_diagnosis, bare_context):
        planner = RecoveryPlanner([_AlwaysApplies("s1", 0.5)])
        recs = planner.recommend(healthy_diagnosis, bare_context)
        assert len(recs) == 1
        assert recs[0].strategy_name == "s1"

    def test_inapplicable_strategy_is_excluded(self, healthy_diagnosis, bare_context):
        planner = RecoveryPlanner([_AlwaysApplies("s1", 0.5, applies=False)])
        assert planner.recommend(healthy_diagnosis, bare_context) == []

    def test_strategy_returning_none_is_excluded(self, healthy_diagnosis, bare_context):
        planner = RecoveryPlanner([_ReturnsNone()])
        assert planner.recommend(healthy_diagnosis, bare_context) == []

    def test_results_sorted_by_priority_descending(self, healthy_diagnosis, bare_context):
        planner = RecoveryPlanner(
            [
                _AlwaysApplies("low", 0.2),
                _AlwaysApplies("high", 0.9),
                _AlwaysApplies("mid", 0.5),
            ]
        )
        recs = planner.recommend(healthy_diagnosis, bare_context)
        assert [r.strategy_name for r in recs] == ["high", "mid", "low"]

    def test_degraded_diagnosis_returns_empty_by_default(self, degraded_diagnosis, bare_context):
        planner = RecoveryPlanner([_AlwaysApplies("s1", 0.9)])
        assert planner.recommend(degraded_diagnosis, bare_context) == []

    def test_degraded_diagnosis_with_allow_degraded_true(self, degraded_diagnosis, bare_context):
        planner = RecoveryPlanner([_AlwaysApplies("s1", 0.9)])
        recs = planner.recommend(degraded_diagnosis, bare_context, allow_degraded=True)
        assert len(recs) == 1

    def test_broken_applies_to_is_skipped_not_raised(self, healthy_diagnosis, bare_context, caplog):
        planner = RecoveryPlanner([_RaisesOnApplicable(), _AlwaysApplies("ok", 0.5)])
        with caplog.at_level(logging.WARNING, logger="qml_observer.recovery"):
            recs = planner.recommend(healthy_diagnosis, bare_context)
        assert len(recs) == 1
        assert recs[0].strategy_name == "ok"
        assert any("raised while proposing" in r.message for r in caplog.records)

    def test_broken_propose_is_skipped_not_raised(self, healthy_diagnosis, bare_context, caplog):
        planner = RecoveryPlanner([_RaisesOnPropose(), _AlwaysApplies("ok", 0.5)])
        with caplog.at_level(logging.WARNING, logger="qml_observer.recovery"):
            recs = planner.recommend(healthy_diagnosis, bare_context)
        assert len(recs) == 1
        assert recs[0].strategy_name == "ok"

    def test_training_never_interrupted_by_all_broken_strategies(
        self, healthy_diagnosis, bare_context
    ):
        planner = RecoveryPlanner([_RaisesOnApplicable(), _RaisesOnPropose()])
        assert planner.recommend(healthy_diagnosis, bare_context) == []  # must not raise
