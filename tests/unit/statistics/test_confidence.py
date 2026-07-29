"""Unit tests for qml_observer.statistics.confidence."""

import math

import numpy as np
import pytest

from qml_observer.schemas.gradient import summarize_gradient
from qml_observer.statistics.confidence import (
    _inverse_normal_cdf,
    _z_score,
    attach_gradient_norm_ci,
    bootstrap_gradient_norm_ci,
    estimate_gradient_norm_ci,
)


class TestInverseNormalCdf:
    def test_median_is_zero(self):
        assert _inverse_normal_cdf(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_known_quantiles(self):
        # Standard normal 97.5th percentile ~= 1.959964
        assert _inverse_normal_cdf(0.975) == pytest.approx(1.959964, abs=1e-5)
        assert _inverse_normal_cdf(0.025) == pytest.approx(-1.959964, abs=1e-5)

    def test_extreme_tail(self):
        # Regression check for the small-p branch of Acklam's algorithm.
        assert _inverse_normal_cdf(0.001) == pytest.approx(-3.09023, abs=1e-4)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            _inverse_normal_cdf(0.0)
        with pytest.raises(ValueError):
            _inverse_normal_cdf(1.0)


class TestZScore:
    def test_95_percent(self):
        assert _z_score(0.95) == pytest.approx(1.959964, abs=1e-5)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            _z_score(1.0)
        with pytest.raises(ValueError):
            _z_score(0.0)


class TestEstimateGradientNormCi:
    def test_basic_interval(self):
        lower, upper = estimate_gradient_norm_ci(norm_l2=1.0, se=0.1, confidence=0.95)
        z = 1.959964
        assert lower == pytest.approx(1.0 - z * 0.1, abs=1e-4)
        assert upper == pytest.approx(1.0 + z * 0.1, abs=1e-4)

    def test_lower_bound_clamped_at_zero(self):
        lower, upper = estimate_gradient_norm_ci(norm_l2=0.01, se=1.0, confidence=0.95)
        assert lower == 0.0
        assert upper > 0.0

    def test_zero_se_gives_zero_width_interval(self):
        lower, upper = estimate_gradient_norm_ci(norm_l2=0.5, se=0.0)
        assert lower == pytest.approx(0.5)
        assert upper == pytest.approx(0.5)

    def test_negative_se_raises(self):
        with pytest.raises(ValueError, match="se"):
            estimate_gradient_norm_ci(norm_l2=1.0, se=-0.1)

    def test_nan_se_yields_nan_interval(self):
        lower, upper = estimate_gradient_norm_ci(norm_l2=1.0, se=float("nan"))
        assert math.isnan(lower)
        assert math.isnan(upper)

    def test_nan_norm_yields_nan_interval(self):
        lower, upper = estimate_gradient_norm_ci(norm_l2=float("nan"), se=0.1)
        assert math.isnan(lower)
        assert math.isnan(upper)

    def test_wider_interval_for_higher_confidence(self):
        lo95, hi95 = estimate_gradient_norm_ci(1.0, 0.1, confidence=0.95)
        lo99, hi99 = estimate_gradient_norm_ci(1.0, 0.1, confidence=0.99)
        assert (hi99 - lo99) > (hi95 - lo95)

    def test_infinite_norm_handled(self):
        lower, upper = estimate_gradient_norm_ci(norm_l2=float("inf"), se=1.0)
        assert lower == 0.0
        assert upper == float("inf")


class TestBootstrapGradientNormCi:
    def test_returns_interval_around_true_norm(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1.0, size=200)
        true_norm = float(np.linalg.norm(values))
        lower, upper = bootstrap_gradient_norm_ci(values, confidence=0.95, n_resamples=500, seed=1)
        assert lower <= true_norm <= upper or abs(lower - true_norm) < 1.0

    def test_reproducible_with_seed(self):
        values = np.array([0.1, -0.2, 0.3, -0.4, 0.05])
        r1 = bootstrap_gradient_norm_ci(values, n_resamples=200, seed=42)
        r2 = bootstrap_gradient_norm_ci(values, n_resamples=200, seed=42)
        assert r1 == r2

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_gradient_norm_ci(np.array([]))

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            bootstrap_gradient_norm_ci(np.array([1.0, 2.0]), confidence=1.5)

    def test_invalid_n_resamples_raises(self):
        with pytest.raises(ValueError):
            bootstrap_gradient_norm_ci(np.array([1.0, 2.0]), n_resamples=0)

    def test_lower_bound_non_negative(self):
        values = np.array([1e-8, -1e-8, 2e-8])
        lower, _ = bootstrap_gradient_norm_ci(values, n_resamples=100, seed=0)
        assert lower >= 0.0


class TestAttachGradientNormCi:
    def test_with_shots_uses_shot_noise_method(self):
        snap = summarize_gradient(np.array([0.5, -0.3, 0.2]))
        result = attach_gradient_norm_ci(snap, shots=100)
        assert result.ci_method == "shot-noise-analytic"
        assert result.ci_level == 0.95
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower <= result.norm_l2 <= result.ci_upper

    def test_without_shots_uses_parameter_spread_method(self):
        snap = summarize_gradient(np.array([0.5, -0.3, 0.2]))
        result = attach_gradient_norm_ci(snap, shots=None)
        assert result.ci_method == "parameter-spread-analytic"

    def test_zero_shots_falls_back_to_parameter_spread(self):
        snap = summarize_gradient(np.array([0.5, -0.3, 0.2]))
        result = attach_gradient_norm_ci(snap, shots=0)
        assert result.ci_method == "parameter-spread-analytic"

    def test_does_not_mutate_input_snapshot(self):
        snap = summarize_gradient(np.array([0.5, -0.3, 0.2]))
        attach_gradient_norm_ci(snap, shots=100)
        assert snap.ci_lower is None
        assert snap.ci_upper is None

    def test_original_summary_stats_preserved(self):
        snap = summarize_gradient(np.array([0.5, -0.3, 0.2]))
        result = attach_gradient_norm_ci(snap, shots=100)
        assert result.norm_l2 == snap.norm_l2
        assert result.mean_abs == snap.mean_abs
        assert result.variance == snap.variance

    def test_more_shots_narrows_interval(self):
        snap = summarize_gradient(np.array([0.5, -0.3, 0.2, 0.1, -0.15]))
        few_shots = attach_gradient_norm_ci(snap, shots=2)
        many_shots = attach_gradient_norm_ci(snap, shots=10_000)
        few_width = few_shots.ci_upper - few_shots.ci_lower
        many_width = many_shots.ci_upper - many_shots.ci_lower
        assert many_width < few_width
