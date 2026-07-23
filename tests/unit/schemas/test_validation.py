"""Unit tests for qml_observer.schemas._validation."""

import math

import numpy as np
import pytest

from qml_observer.schemas._validation import (
    check_finite_number,
    check_non_empty_str,
    check_non_negative_int,
    check_non_negative_number,
    check_range,
    check_str_list,
    check_type,
)


class TestCheckType:
    def test_passes_for_matching_type(self):
        check_type(5, int, "x")
        check_type("hi", str, "x")
        check_type(1.0, (int, float), "x")

    def test_raises_for_mismatched_type(self):
        with pytest.raises(TypeError, match="x must be"):
            check_type("hi", int, "x")


class TestCheckNonEmptyStr:
    def test_passes_for_non_empty(self):
        check_non_empty_str("hello", "x")

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_raises_for_blank(self, value):
        with pytest.raises(ValueError, match="non-empty string"):
            check_non_empty_str(value, "x")

    def test_raises_for_non_str(self):
        with pytest.raises(TypeError):
            check_non_empty_str(5, "x")  # type: ignore[arg-type]


class TestCheckNonNegativeInt:
    def test_none_passes(self):
        check_non_negative_int(None, "x")

    @pytest.mark.parametrize("value", [0, 1, 1000])
    def test_passes_for_non_negative(self, value):
        check_non_negative_int(value, "x")

    def test_raises_for_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            check_non_negative_int(-1, "x")

    def test_raises_for_bool(self):
        with pytest.raises(TypeError, match="bool"):
            check_non_negative_int(True, "x")

    def test_raises_for_float(self):
        with pytest.raises(TypeError):
            check_non_negative_int(1.5, "x")  # type: ignore[arg-type]


class TestCheckNonNegativeNumber:
    def test_none_passes(self):
        check_non_negative_number(None, "x")

    @pytest.mark.parametrize("value", [0, 0.0, 5, 3.14])
    def test_passes_for_non_negative(self, value):
        check_non_negative_number(value, "x")

    def test_raises_for_negative(self):
        with pytest.raises(ValueError, match=">= 0"):
            check_non_negative_number(-0.001, "x")

    def test_nan_is_tolerated(self):
        """NaN must never be rejected here (addendum §7)."""
        check_non_negative_number(float("nan"), "x")

    def test_inf_is_tolerated(self):
        check_non_negative_number(float("inf"), "x")

    def test_raises_for_bool(self):
        with pytest.raises(TypeError, match="bool"):
            check_non_negative_number(True, "x")

    def test_raises_for_non_number(self):
        with pytest.raises(TypeError):
            check_non_negative_number("5", "x")  # type: ignore[arg-type]


class TestCheckFiniteNumber:
    def test_none_passes(self):
        check_finite_number(None, "x")

    @pytest.mark.parametrize("value", [0, 0.0, -5, 3.14])
    def test_passes_for_finite_numbers(self, value):
        check_finite_number(value, "x")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_raises_for_non_finite(self, value):
        with pytest.raises(ValueError, match="finite"):
            check_finite_number(value, "x")

    def test_raises_for_bool(self):
        with pytest.raises(TypeError, match="bool"):
            check_finite_number(True, "x")

    def test_raises_for_non_number(self):
        with pytest.raises(TypeError):
            check_finite_number("5", "x")  # type: ignore[arg-type]


class TestCheckRange:
    def test_passes_within_range(self):
        check_range(0.5, 0.0, 1.0, "x")

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_boundaries_are_inclusive(self, value):
        check_range(value, 0.0, 1.0, "x")

    @pytest.mark.parametrize("value", [-0.01, 1.01, 100])
    def test_raises_outside_range(self, value):
        with pytest.raises(ValueError, match=r"must be in \[0.0, 1.0\]"):
            check_range(value, 0.0, 1.0, "x")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_raises_for_non_finite(self, value):
        with pytest.raises(ValueError, match="finite"):
            check_range(value, 0.0, 1.0, "x")

    def test_raises_for_bool(self):
        with pytest.raises(TypeError, match="bool"):
            check_range(True, 0.0, 1.0, "x")


class TestCheckStrList:
    def test_passes_for_list_of_str(self):
        check_str_list([], "x")
        check_str_list(["a", "b"], "x")

    def test_raises_for_non_list(self):
        with pytest.raises(TypeError, match="must be a list"):
            check_str_list("not a list", "x")

    def test_raises_for_non_str_item(self):
        with pytest.raises(TypeError, match=r"x\[1\] must be a str"):
            check_str_list(["ok", 5], "x")


def test_math_isnan_sanity():
    """Sanity check that our NaN-tolerance assumption about numpy floats holds."""
    arr = np.array([float("nan")])
    assert math.isnan(float(arr[0]))
