"""Unit tests for qml_observer.recovery.evaluation.RecoveryEvaluator."""

from __future__ import annotations

import pytest

from qml_observer.recovery.evaluation import RecoveryEvaluator


class TestConstruction:
    def test_default_threshold(self):
        assert RecoveryEvaluator()._confidence_improvement_threshold == 0.1

    def test_custom_threshold(self):
        assert RecoveryEvaluator(0.2)._confidence_improvement_threshold == 0.2

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_out_of_range_threshold_raises(self, bad):
        with pytest.raises(ValueError, match="confidence_improvement_threshold"):
            RecoveryEvaluator(bad)


class TestDegradedHandling:
    def test_before_degraded_is_inconclusive(self, degraded_diagnosis, healthy_diagnosis):
        result = RecoveryEvaluator().evaluate("s", degraded_diagnosis, healthy_diagnosis)
        assert result.conclusive is False
        assert result.improved is False
        assert "before" in result.summary.lower()

    def test_after_degraded_is_inconclusive(self, healthy_diagnosis, degraded_diagnosis):
        result = RecoveryEvaluator().evaluate("s", healthy_diagnosis, degraded_diagnosis)
        assert result.conclusive is False
        assert result.improved is False
        assert "after" in result.summary.lower()

    def test_both_degraded_is_inconclusive(self, degraded_diagnosis):
        result = RecoveryEvaluator().evaluate("s", degraded_diagnosis, degraded_diagnosis)
        assert result.conclusive is False


class TestReachingGoodState:
    def test_bad_to_healthy_is_improved(self, barren_plateau_diagnosis, healthy_diagnosis):
        result = RecoveryEvaluator().evaluate("s", barren_plateau_diagnosis, healthy_diagnosis)
        assert result.conclusive is True
        assert result.improved is True

    def test_bad_to_converged_is_improved(self, diagnosis_factory, barren_plateau_diagnosis):
        from qml_observer.schemas.diagnosis import IssueType

        converged = diagnosis_factory(issue=IssueType.CONVERGED, confidence=0.9, severity="info")
        result = RecoveryEvaluator().evaluate("s", barren_plateau_diagnosis, converged)
        assert result.improved is True


class TestRegression:
    def test_healthy_to_bad_is_regression(self, healthy_diagnosis, barren_plateau_diagnosis):
        result = RecoveryEvaluator().evaluate("s", healthy_diagnosis, barren_plateau_diagnosis)
        assert result.conclusive is True
        assert result.improved is False
        assert "regress" in result.summary.lower()


class TestSameIssue:
    def test_severity_drop_is_improved(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        before = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.7, severity="critical")
        after = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.7, severity="warning")
        result = RecoveryEvaluator().evaluate("s", before, after)
        assert result.improved is True
        assert result.conclusive is True

    def test_severity_rise_is_not_improved(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        before = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.7, severity="warning")
        after = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.7, severity="critical")
        result = RecoveryEvaluator().evaluate("s", before, after)
        assert result.improved is False

    def test_confidence_drop_above_threshold_is_improved(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        before = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.9, severity="warning")
        after = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.5, severity="warning")
        result = RecoveryEvaluator(confidence_improvement_threshold=0.1).evaluate(
            "s", before, after
        )
        assert result.improved is True

    def test_confidence_drop_below_threshold_is_unchanged(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        before = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.55, severity="warning")
        after = diagnosis_factory(issue=IssueType.STAGNATION, confidence=0.5, severity="warning")
        result = RecoveryEvaluator(confidence_improvement_threshold=0.2).evaluate(
            "s", before, after
        )
        assert result.improved is False
        assert result.conclusive is True

    def test_identical_diagnosis_is_unchanged(self, stagnation_diagnosis):
        result = RecoveryEvaluator().evaluate("s", stagnation_diagnosis, stagnation_diagnosis)
        assert result.improved is False
        assert result.conclusive is True


class TestDifferentIssue:
    def test_different_issue_lower_severity_is_improved(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        before = diagnosis_factory(
            issue=IssueType.POSSIBLE_BARREN_PLATEAU, confidence=0.9, severity="critical"
        )
        after = diagnosis_factory(
            issue=IssueType.NOISE_DOMINATED, confidence=0.6, severity="warning"
        )
        result = RecoveryEvaluator().evaluate("s", before, after)
        assert result.improved is True
        assert result.conclusive is True

    def test_different_issue_same_severity_is_not_improved(self, diagnosis_factory):
        from qml_observer.schemas.diagnosis import IssueType

        before = diagnosis_factory(
            issue=IssueType.POSSIBLE_BARREN_PLATEAU, confidence=0.9, severity="critical"
        )
        after = diagnosis_factory(issue=IssueType.UNSTABLE, confidence=0.6, severity="critical")
        result = RecoveryEvaluator().evaluate("s", before, after)
        assert result.improved is False


class TestShouldKeep:
    def test_keeps_conclusive_improved(self, barren_plateau_diagnosis, healthy_diagnosis):
        evaluator = RecoveryEvaluator()
        result = evaluator.evaluate("s", barren_plateau_diagnosis, healthy_diagnosis)
        assert evaluator.should_keep(result) is True

    def test_does_not_keep_inconclusive(self, degraded_diagnosis, healthy_diagnosis):
        evaluator = RecoveryEvaluator()
        result = evaluator.evaluate("s", degraded_diagnosis, healthy_diagnosis)
        assert evaluator.should_keep(result) is False

    def test_does_not_keep_conclusive_not_improved(
        self, healthy_diagnosis, barren_plateau_diagnosis
    ):
        evaluator = RecoveryEvaluator()
        result = evaluator.evaluate("s", healthy_diagnosis, barren_plateau_diagnosis)
        assert evaluator.should_keep(result) is False


class TestResultShape:
    def test_strategy_name_preserved(self, healthy_diagnosis, barren_plateau_diagnosis):
        result = RecoveryEvaluator().evaluate(
            "my_strategy", healthy_diagnosis, barren_plateau_diagnosis
        )
        assert result.strategy_name == "my_strategy"

    def test_before_after_preserved(self, healthy_diagnosis, barren_plateau_diagnosis):
        result = RecoveryEvaluator().evaluate("s", healthy_diagnosis, barren_plateau_diagnosis)
        assert result.before is healthy_diagnosis
        assert result.after is barren_plateau_diagnosis
