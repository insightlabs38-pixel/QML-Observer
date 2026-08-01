"""Unit tests for qml_observer.recovery.executor.RecoveryExecutor."""

from __future__ import annotations

import logging

from qml_observer.recovery.base import RecoveryRecommendation
from qml_observer.recovery.executor import RecoveryExecutor


class TestApply:
    def test_no_hook_name_is_manual_only(self):
        rec = RecoveryRecommendation(strategy_name="s", description="do it manually", priority=0.5)
        outcome = RecoveryExecutor().apply(rec, object())
        assert outcome.applied is False
        assert "manually" in outcome.message
        assert outcome.strategy_name == "s"

    def test_missing_hook_reports_not_applied(self):
        rec = RecoveryRecommendation(
            strategy_name="s", description="x", priority=0.5, hook_name="does_not_exist"
        )

        class EmptyState:
            pass

        outcome = RecoveryExecutor().apply(rec, EmptyState())
        assert outcome.applied is False
        assert "does_not_exist" in outcome.message

    def test_non_callable_attribute_is_treated_as_missing(self):
        rec = RecoveryRecommendation(
            strategy_name="s", description="x", priority=0.5, hook_name="set_learning_rate"
        )

        class BadState:
            set_learning_rate = "not-a-method"

        outcome = RecoveryExecutor().apply(rec, BadState())
        assert outcome.applied is False

    def test_matching_hook_is_called_with_parameters(self):
        rec = RecoveryRecommendation(
            strategy_name="s",
            description="x",
            priority=0.5,
            parameters={"learning_rate": 0.001},
            hook_name="set_learning_rate",
        )

        calls = []

        class GoodState:
            def set_learning_rate(self, learning_rate):
                calls.append(learning_rate)

        outcome = RecoveryExecutor().apply(rec, GoodState())
        assert outcome.applied is True
        assert calls == [0.001]
        assert "set_learning_rate" in outcome.message

    def test_hook_with_no_parameters_is_called_with_none(self):
        rec = RecoveryRecommendation(
            strategy_name="s", description="x", priority=0.5, hook_name="reinitialize_parameters"
        )

        calls = []

        class GoodState:
            def reinitialize_parameters(self):
                calls.append(True)

        outcome = RecoveryExecutor().apply(rec, GoodState())
        assert outcome.applied is True
        assert calls == [True]

    def test_hook_raising_is_caught_fail_open(self, caplog):
        rec = RecoveryRecommendation(
            strategy_name="s", description="fallback text", priority=0.5, hook_name="set_shots"
        )

        class BrokenState:
            def set_shots(self, **kwargs):
                raise ValueError("optimizer not attached")

        with caplog.at_level(logging.WARNING, logger="qml_observer.recovery"):
            outcome = RecoveryExecutor().apply(rec, BrokenState())

        assert outcome.applied is False
        assert "ValueError" in outcome.message
        assert "fallback text" in outcome.message
        assert any("applying" in r.message for r in caplog.records)

    def test_hook_raising_never_propagates(self):
        rec = RecoveryRecommendation(
            strategy_name="s", description="x", priority=0.5, hook_name="set_shots"
        )

        class BrokenState:
            def set_shots(self, **kwargs):
                raise RuntimeError("nope")

        # Must not raise.
        outcome = RecoveryExecutor().apply(rec, BrokenState())
        assert outcome.applied is False
