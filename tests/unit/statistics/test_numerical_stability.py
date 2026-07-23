"""Numerical stability tests for the statistics engine.

Milestone 3, Issue #24 ("Add numerical stability tests"). Consolidates
the specific edge cases called out in addendum §7 into one place for
traceability, on top of the case-by-case coverage already present in
`test_gradients.py`, `test_loss.py`, and `test_rolling.py`.

Addendum §7 checklist:

1. `RollingWindow` with zero or one observation -- variance/slope must
   return `None`, not raise. (Also covered per-method in
   `test_rolling.py::TestVariance`/`TestSlope`; re-asserted here as a
   single checklist item, plus the insufficient-data-vs-nonfinite
   ordering edge case.)
2. `estimate_gradient_snr` with `gradient_std == 0` -- out of scope for
   this batch. `statistics/snr.py` is part of the Volume IV spec but is
   not one of Milestone 3's first-five issues (#18-#22) or Issue #23;
   it lands with the noise/SNR work (Milestone 9). Not implemented or
   tested here.
3. NaN/Inf loss values from a diverging optimizer -- the statistics
   layer must propagate these as `nan`/`inf` rather than raising;
   classifying them as `IssueType.UNSTABLE` is a detector-layer concern
   (Milestone 4), out of scope here.
4. Empty gradient arrays -- must raise a clear, catchable `ValueError`,
   not an opaque numpy exception.
"""

import math

import numpy as np
import pytest

from qml_observer.statistics.gradients import (
    gradient_norm,
    gradient_percentiles,
    gradient_variance,
    mean_absolute_gradient,
)
from qml_observer.statistics.loss import loss_slope, relative_loss_improvement
from qml_observer.statistics.rolling import RollingWindow


class TestRollingWindowZeroOrOneObservation:
    """Addendum §7, item 1."""

    @pytest.mark.parametrize("method", ["variance", "slope"])
    def test_empty_window_returns_none_not_raises(self, method):
        w = RollingWindow(maxlen=5)
        assert getattr(w, method)() is None

    @pytest.mark.parametrize("method", ["variance", "slope"])
    def test_single_observation_returns_none_not_raises(self, method):
        w = RollingWindow(maxlen=5)
        w.append(1.0)
        assert getattr(w, method)() is None

    def test_single_nonfinite_observation_is_still_none_not_nan(self):
        """Insufficient-data check must run before the non-finite check:
        a lone NaN observation is still "not enough data" (`None`), not
        a computed `nan` result."""
        w = RollingWindow(maxlen=5)
        w.append(float("nan"))
        assert w.variance() is None
        assert w.slope() is None
        # mean() *is* well-defined for a single observation, including NaN.
        assert math.isnan(w.mean())


class TestNanInfLossPropagation:
    """Addendum §7, item 3 -- must propagate, never raise."""

    def test_loss_slope_with_nan_returns_nan(self):
        assert math.isnan(loss_slope([1.0, float("nan"), 0.5]))

    def test_loss_slope_with_inf_returns_nan(self):
        assert math.isnan(loss_slope([1.0, float("inf")]))

    def test_relative_loss_improvement_with_nan_returns_nan(self):
        assert math.isnan(relative_loss_improvement([float("nan"), 1.0]))

    def test_relative_loss_improvement_with_inf_baseline_is_defined_or_nan(self):
        # inf - finite = inf; inf / inf = nan (numpy semantics); assert
        # this does not raise, whatever value it settles on.
        result = relative_loss_improvement([float("inf"), 1.0])
        assert isinstance(result, float)


class TestEmptyGradientArraysRaiseClearErrors:
    """Addendum §7, item 4 -- clear ValueError, not an opaque numpy error."""

    @pytest.mark.parametrize(
        "func",
        [gradient_norm, mean_absolute_gradient, gradient_variance, gradient_percentiles],
    )
    def test_empty_array_raises_value_error_with_clear_message(self, func):
        with pytest.raises(ValueError, match="empty gradient array"):
            func(np.array([]))


class TestGradientStatsToleratesNonFiniteValues:
    """Non-finite gradient values are signal, not an error, across every
    gradient statistic (complements the per-function tests in
    test_gradients.py with an all-NaN / all-Inf sweep)."""

    @pytest.mark.parametrize(
        "func",
        [gradient_norm, mean_absolute_gradient, gradient_variance],
    )
    def test_all_nan_array_returns_nan_without_raising(self, func):
        result = func(np.array([float("nan"), float("nan")]))
        assert math.isnan(result)

    def test_percentiles_with_nan_present_does_not_raise(self):
        # numpy propagates NaN through percentile computation; the
        # important contract here is that it does not raise.
        result = gradient_percentiles(np.array([1.0, float("nan"), 3.0]))
        assert set(result.keys()) == {1, 10, 50, 90, 99}
