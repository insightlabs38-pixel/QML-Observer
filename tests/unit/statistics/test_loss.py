"""Unit tests for qml_observer.statistics.loss."""

import math

import pytest

from qml_observer.statistics.loss import (
    is_loss_stagnant,
    loss_slope,
    relative_loss_improvement,
)


class TestLossSlope:
    def test_decreasing_loss_has_negative_slope(self):
        losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        assert loss_slope(losses) == pytest.approx(-0.2)

    def test_flat_loss_has_zero_slope(self):
        losses = [0.5, 0.5, 0.5, 0.5]
        assert loss_slope(losses) == pytest.approx(0.0, abs=1e-12)

    def test_increasing_loss_has_positive_slope(self):
        losses = [0.1, 0.2, 0.3]
        assert loss_slope(losses) == pytest.approx(0.1)

    def test_fewer_than_two_values_raises(self):
        with pytest.raises(ValueError, match="at least 2 loss values"):
            loss_slope([0.5])
        with pytest.raises(ValueError, match="at least 2 loss values"):
            loss_slope([])

    def test_nan_in_losses_yields_nan_slope(self):
        assert math.isnan(loss_slope([1.0, float("nan"), 0.5]))

    def test_inf_in_losses_yields_nan_slope(self):
        assert math.isnan(loss_slope([1.0, float("inf"), 0.5]))


class TestRelativeLossImprovement:
    def test_improvement_from_positive_baseline(self):
        # loss halved: (1.0 - 0.5) / 1.0 = 0.5
        assert relative_loss_improvement([1.0, 0.7, 0.5]) == pytest.approx(0.5)

    def test_worsening_loss_is_negative(self):
        assert relative_loss_improvement([1.0, 1.5]) == pytest.approx(-0.5)

    def test_no_change_is_zero(self):
        assert relative_loss_improvement([0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_zero_baseline_no_change_returns_zero(self):
        assert relative_loss_improvement([0.0, 0.0]) == 0.0

    def test_zero_baseline_with_change_returns_signed_inf(self):
        assert relative_loss_improvement([0.0, -1.0]) == float("inf")
        assert relative_loss_improvement([0.0, 1.0]) == float("-inf")

    def test_fewer_than_two_values_raises(self):
        with pytest.raises(ValueError, match="at least 2 loss values"):
            relative_loss_improvement([1.0])

    def test_nan_loss_yields_nan(self):
        assert math.isnan(relative_loss_improvement([float("nan"), 1.0]))
        assert math.isnan(relative_loss_improvement([1.0, float("nan")]))


class TestIsLossStagnant:
    def test_small_improvement_is_stagnant(self):
        losses = [0.5000, 0.5001]
        assert is_loss_stagnant(losses, threshold=1e-2) is True

    def test_large_improvement_is_not_stagnant(self):
        losses = [1.0, 0.5]
        assert is_loss_stagnant(losses, threshold=1e-2) is False

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            is_loss_stagnant([1.0, 0.9], threshold=-0.1)

    def test_undefined_improvement_is_not_stagnant(self):
        """NaN improvement must never be silently reported as stagnation."""
        assert is_loss_stagnant([float("nan"), 1.0], threshold=1.0) is False
