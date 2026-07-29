"""Unit tests for qml_observer.schemas.gradient."""

import numpy as np
import pytest

from qml_observer.schemas.gradient import GradientSnapshot, summarize_gradient


class TestSummarizeGradient:
    def test_basic_statistics_match_numpy(self):
        grads = np.array([1e-3, -2e-3, 5e-4, -1e-4])
        snap = summarize_gradient(grads, method="parameter-shift")

        assert snap.method == "parameter-shift"
        assert snap.snr is None
        assert snap.uncertainty is None
        assert snap.norm_l2 == pytest.approx(float(np.linalg.norm(grads)))
        assert snap.mean_abs == pytest.approx(float(np.mean(np.abs(grads))))
        assert snap.variance == pytest.approx(float(np.var(grads)))
        assert snap.min_value == pytest.approx(float(grads.min()))
        assert snap.max_value == pytest.approx(float(grads.max()))
        assert snap.median_abs == pytest.approx(float(np.median(np.abs(grads))))
        assert np.array_equal(snap.values, grads)

    def test_multi_dimensional_gradient_is_flattened_for_stats(self):
        grads = np.array([[1e-5, 2e-5], [3e-5, 4e-5]])
        snap = summarize_gradient(grads)

        assert snap.mean_abs == pytest.approx(float(np.mean(np.abs(grads))))
        assert snap.norm_l2 == pytest.approx(float(np.linalg.norm(grads.ravel())))
        # original (unflattened) shape is preserved in `values`
        assert snap.values.shape == (2, 2)

    def test_list_input_is_coerced_to_ndarray(self):
        snap = summarize_gradient([0.1, -0.2, 0.3])
        assert isinstance(snap.values, np.ndarray)

    def test_keep_values_false_drops_raw_array(self):
        snap = summarize_gradient(np.array([0.1, 0.2]), keep_values=False)
        assert snap.values is None
        # stats are still computed
        assert snap.mean_abs == pytest.approx(0.15)

    def test_empty_array_raises_clear_error(self):
        """Per addendum §7: must be a clear, catchable error, not an
        opaque numpy exception."""
        with pytest.raises(ValueError, match="empty gradient array"):
            summarize_gradient(np.array([]))

    def test_nan_gradient_values_propagate_without_raising(self):
        """NaN gradients are a meaningful diverging-training signal for the
        detector layer, not a construction-time error."""
        snap = summarize_gradient(np.array([float("nan"), 1.0]))
        assert snap.norm_l2 != snap.norm_l2  # NaN


class TestGradientSnapshotValidation:
    def _kwargs(self, **overrides):
        base = dict(
            values=None,
            norm_l2=1.0,
            mean_abs=0.5,
            variance=0.01,
            min_value=-1.0,
            max_value=1.0,
            median_abs=0.4,
        )
        base.update(overrides)
        return base

    def test_valid_manual_construction(self):
        snap = GradientSnapshot(**self._kwargs())
        assert snap.norm_l2 == 1.0

    def test_negative_norm_l2_raises(self):
        with pytest.raises(ValueError, match="norm_l2"):
            GradientSnapshot(**self._kwargs(norm_l2=-1.0))

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError, match="variance"):
            GradientSnapshot(**self._kwargs(variance=-0.1))

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="min_value"):
            GradientSnapshot(**self._kwargs(min_value=5.0, max_value=1.0))

    def test_nan_min_or_max_skips_ordering_check(self):
        GradientSnapshot(**self._kwargs(min_value=float("nan"), max_value=1.0))
        GradientSnapshot(**self._kwargs(min_value=-1.0, max_value=float("nan")))

    def test_non_ndarray_values_raises(self):
        with pytest.raises(TypeError, match="values"):
            GradientSnapshot(**self._kwargs(values=[1, 2, 3]))

    def test_negative_snr_raises(self):
        with pytest.raises(ValueError, match="snr"):
            GradientSnapshot(**self._kwargs(snr=-0.5))

    def test_nan_snr_is_tolerated(self):
        GradientSnapshot(**self._kwargs(snr=float("nan")))

    def test_negative_uncertainty_raises(self):
        with pytest.raises(ValueError, match="uncertainty"):
            GradientSnapshot(**self._kwargs(uncertainty=-0.01))

    def test_valid_ci_fields(self):
        snap = GradientSnapshot(
            **self._kwargs(ci_lower=0.5, ci_upper=1.5, ci_level=0.95, ci_method="analytic")
        )
        assert snap.ci_lower == 0.5
        assert snap.ci_upper == 1.5
        assert snap.ci_level == 0.95
        assert snap.ci_method == "analytic"

    def test_negative_ci_lower_raises(self):
        with pytest.raises(ValueError, match="ci_lower"):
            GradientSnapshot(**self._kwargs(ci_lower=-0.1, ci_upper=1.0))

    def test_negative_ci_upper_raises(self):
        with pytest.raises(ValueError, match="ci_upper"):
            GradientSnapshot(**self._kwargs(ci_lower=0.1, ci_upper=-1.0))

    def test_ci_lower_greater_than_ci_upper_raises(self):
        with pytest.raises(ValueError, match="ci_lower"):
            GradientSnapshot(**self._kwargs(ci_lower=2.0, ci_upper=1.0))

    def test_nan_ci_bounds_skip_ordering_check(self):
        GradientSnapshot(**self._kwargs(ci_lower=float("nan"), ci_upper=1.0))

    def test_ci_level_out_of_range_raises(self):
        with pytest.raises(ValueError, match="ci_level"):
            GradientSnapshot(**self._kwargs(ci_level=1.5))
        with pytest.raises(ValueError, match="ci_level"):
            GradientSnapshot(**self._kwargs(ci_level=0.0))

    def test_ci_method_must_be_str(self):
        with pytest.raises(TypeError, match="ci_method"):
            GradientSnapshot(**self._kwargs(ci_method=123))

    def test_ci_fields_default_to_none(self):
        snap = GradientSnapshot(**self._kwargs())
        assert snap.ci_lower is None
        assert snap.ci_upper is None
        assert snap.ci_level is None
        assert snap.ci_method is None
