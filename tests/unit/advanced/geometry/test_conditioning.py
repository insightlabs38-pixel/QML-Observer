"""Unit tests for qml_observer.advanced.geometry.conditioning."""

import numpy as np
import pytest

from qml_observer.advanced.geometry.conditioning import (
    ConditioningResult,
    effective_rank,
    qfim_condition_number,
    summarize_conditioning,
)


def diag_qfim(*values: float) -> np.ndarray:
    return np.diag(np.array(values, dtype=float))


class TestQfimConditionNumber:
    def test_identity_like_qfim_has_condition_number_one(self):
        fim = diag_qfim(1.0, 1.0, 1.0)
        assert qfim_condition_number(fim) == pytest.approx(1.0)

    def test_anisotropic_qfim(self):
        fim = diag_qfim(4.0, 1.0)
        assert qfim_condition_number(fim) == pytest.approx(4.0)

    def test_near_singular_qfim_is_inf(self):
        fim = diag_qfim(1.0, 1e-14)
        assert qfim_condition_number(fim) == float("inf")

    def test_all_zero_qfim_is_inf(self):
        fim = diag_qfim(0.0, 0.0)
        assert qfim_condition_number(fim) == float("inf")

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            qfim_condition_number(np.zeros((2, 3)))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            qfim_condition_number(np.zeros((0, 0)))


class TestEffectiveRank:
    def test_full_rank_qfim(self):
        fim = diag_qfim(1.0, 0.9, 1.1)
        assert effective_rank(fim) == 3

    def test_one_near_zero_eigenvalue_drops_rank(self):
        fim = diag_qfim(1.0, 1.0, 1e-10)
        assert effective_rank(fim) == 2

    def test_all_zero_qfim_has_rank_zero(self):
        assert effective_rank(diag_qfim(0.0, 0.0)) == 0

    def test_threshold_is_relative_to_largest_eigenvalue(self):
        fim = diag_qfim(100.0, 1.0)
        # 1.0 / 100.0 = 0.01, right at a threshold of 0.01 -> counted.
        assert effective_rank(fim, threshold=0.005) == 2
        assert effective_rank(fim, threshold=0.5) == 1


class TestSummarizeConditioning:
    def test_matches_individual_calls(self):
        fim = diag_qfim(4.0, 1.0, 1e-10)
        result = summarize_conditioning(fim)
        assert isinstance(result, ConditioningResult)
        assert result.condition_number == qfim_condition_number(fim)
        assert result.effective_rank == effective_rank(fim)
        assert result.n_parameters == 3
        assert result.eigenvalues[0] == pytest.approx(4.0)

    def test_eigenvalues_are_descending(self):
        fim = diag_qfim(1.0, 5.0, 2.0)
        result = summarize_conditioning(fim)
        assert list(result.eigenvalues) == sorted(result.eigenvalues, reverse=True)
