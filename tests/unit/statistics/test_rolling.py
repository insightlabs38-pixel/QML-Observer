"""Unit tests for qml_observer.statistics.rolling.RollingWindow."""

import math

import numpy as np
import pytest

from qml_observer.statistics.rolling import RollingWindow


class TestConstruction:
    def test_valid_maxlen(self):
        w = RollingWindow(maxlen=10)
        assert len(w) == 0

    def test_non_int_maxlen_raises(self):
        with pytest.raises(TypeError, match="maxlen"):
            RollingWindow(maxlen=10.0)

    def test_bool_maxlen_raises(self):
        with pytest.raises(TypeError, match="maxlen"):
            RollingWindow(maxlen=True)

    def test_zero_maxlen_raises(self):
        with pytest.raises(ValueError, match="maxlen"):
            RollingWindow(maxlen=0)

    def test_negative_maxlen_raises(self):
        with pytest.raises(ValueError, match="maxlen"):
            RollingWindow(maxlen=-5)


class TestAppendAndValues:
    def test_append_and_values_order(self):
        w = RollingWindow(maxlen=3)
        for v in (1.0, 2.0, 3.0):
            w.append(v)
        assert w.values() == [1.0, 2.0, 3.0]
        assert len(w) == 3

    def test_eviction_when_full(self):
        w = RollingWindow(maxlen=3)
        for v in (1.0, 2.0, 3.0, 4.0):
            w.append(v)
        assert w.values() == [2.0, 3.0, 4.0]
        assert len(w) == 3

    def test_int_values_are_coerced_to_float(self):
        w = RollingWindow(maxlen=3)
        w.append(1)
        assert w.values() == [1.0]
        assert isinstance(w.values()[0], float)

    def test_non_numeric_value_raises(self):
        w = RollingWindow(maxlen=3)
        with pytest.raises(TypeError, match="value"):
            w.append("nope")

    def test_bool_value_raises(self):
        w = RollingWindow(maxlen=3)
        with pytest.raises(TypeError, match="value"):
            w.append(True)


class TestMean:
    def test_empty_window_mean_is_none(self):
        assert RollingWindow(maxlen=5).mean() is None

    def test_mean_matches_numpy(self):
        w = RollingWindow(maxlen=5)
        for v in (1.0, 2.0, 3.0, 4.0):
            w.append(v)
        assert w.mean() == pytest.approx(2.5)

    def test_single_value_mean_is_that_value(self):
        w = RollingWindow(maxlen=5)
        w.append(7.0)
        assert w.mean() == pytest.approx(7.0)


class TestVariance:
    def test_empty_window_variance_is_none(self):
        assert RollingWindow(maxlen=5).variance() is None

    def test_single_value_variance_is_none(self):
        """Undefined for 1 observation -- must be None, not 0.0 (addendum §7)."""
        w = RollingWindow(maxlen=5)
        w.append(7.0)
        assert w.variance() is None

    def test_variance_matches_numpy(self):
        w = RollingWindow(maxlen=10)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            w.append(v)
        assert w.variance() == pytest.approx(float(np.var(values)))

    def test_variance_after_eviction_matches_numpy(self):
        w = RollingWindow(maxlen=3)
        for v in (10.0, 1.0, 2.0, 3.0):
            w.append(v)
        assert w.values() == [1.0, 2.0, 3.0]
        assert w.variance() == pytest.approx(float(np.var([1.0, 2.0, 3.0])))

    def test_variance_never_negative_despite_fp_noise(self):
        w = RollingWindow(maxlen=1000)
        for _ in range(1000):
            w.append(1e15)
        assert w.variance() >= 0.0


class TestSlope:
    def test_empty_window_slope_is_none(self):
        assert RollingWindow(maxlen=5).slope() is None

    def test_single_value_slope_is_none(self):
        w = RollingWindow(maxlen=5)
        w.append(1.0)
        assert w.slope() is None

    def test_decreasing_values_have_negative_slope(self):
        w = RollingWindow(maxlen=5)
        for v in (1.0, 0.8, 0.6, 0.4, 0.2):
            w.append(v)
        assert w.slope() == pytest.approx(-0.2)

    def test_slope_reflects_only_current_window_after_eviction(self):
        w = RollingWindow(maxlen=2)
        w.append(100.0)  # will be evicted
        w.append(1.0)
        w.append(2.0)
        # window is now [1.0, 2.0] -> slope 1.0, not influenced by 100.0
        assert w.slope() == pytest.approx(1.0)


class TestNonFiniteValues:
    """Addendum §7: NaN/Inf are valid entries, and must not permanently
    poison the incremental aggregates once evicted."""

    def test_nan_in_window_makes_mean_nan(self):
        w = RollingWindow(maxlen=5)
        w.append(1.0)
        w.append(float("nan"))
        assert math.isnan(w.mean())

    def test_nan_in_window_makes_variance_nan(self):
        w = RollingWindow(maxlen=5)
        w.append(1.0)
        w.append(2.0)
        w.append(float("nan"))
        assert math.isnan(w.variance())

    def test_nan_in_window_makes_slope_nan(self):
        w = RollingWindow(maxlen=5)
        w.append(1.0)
        w.append(float("nan"))
        assert math.isnan(w.slope())

    def test_mean_recovers_after_nan_is_evicted(self):
        w = RollingWindow(maxlen=2)
        w.append(float("nan"))
        assert math.isnan(w.mean())
        w.append(1.0)
        w.append(2.0)  # NaN has now been evicted out of the window
        assert w.mean() == pytest.approx(1.5)

    def test_variance_recovers_after_nan_is_evicted(self):
        w = RollingWindow(maxlen=2)
        w.append(float("nan"))
        w.append(1.0)
        w.append(2.0)
        w.append(3.0)  # window is [2.0, 3.0]; NaN long evicted
        assert w.variance() == pytest.approx(float(np.var([2.0, 3.0])))

    def test_inf_recovers_after_eviction(self):
        w = RollingWindow(maxlen=2)
        w.append(float("inf"))
        assert w.mean() == float("inf")
        w.append(1.0)
        w.append(2.0)
        assert w.mean() == pytest.approx(1.5)
        assert w.variance() == pytest.approx(float(np.var([1.0, 2.0])))

    def test_repeated_nan_eviction_cycles_stay_correct(self):
        """Stress the incremental bookkeeping across many NaN in/out cycles."""
        w = RollingWindow(maxlen=3)
        expected_tail = []
        for i in range(50):
            val = float("nan") if i % 4 == 0 else float(i)
            w.append(val)
            expected_tail.append(val)
            expected_tail = expected_tail[-3:]

        if any(math.isnan(v) for v in expected_tail):
            assert math.isnan(w.mean())
        else:
            assert w.mean() == pytest.approx(float(np.mean(expected_tail)))


class TestReset:
    def test_reset_clears_values_and_aggregates(self):
        w = RollingWindow(maxlen=3)
        for v in (1.0, 2.0, 3.0):
            w.append(v)
        w.reset()
        assert len(w) == 0
        assert w.values() == []
        assert w.mean() is None
        assert w.variance() is None
        assert w.slope() is None

    def test_window_usable_after_reset(self):
        w = RollingWindow(maxlen=3)
        w.append(float("nan"))
        w.reset()
        w.append(5.0)
        w.append(6.0)
        assert w.mean() == pytest.approx(5.5)
        assert w.variance() == pytest.approx(float(np.var([5.0, 6.0])))
