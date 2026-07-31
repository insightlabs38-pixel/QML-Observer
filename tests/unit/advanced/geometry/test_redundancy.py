"""Unit tests for qml_observer.advanced.geometry.redundancy."""

import numpy as np
import pytest

from qml_observer.advanced.geometry.redundancy import (
    RedundancyResult,
    detect_redundant_parameters,
)


class TestDetectRedundantParameters:
    def test_full_rank_qfim_has_no_redundancy(self):
        fim = np.diag([1.0, 1.0, 1.0])
        result = detect_redundant_parameters(fim)
        assert isinstance(result, RedundancyResult)
        assert result.redundant_parameter_indices == []
        assert result.null_space_dimension == 0
        assert result.effective_rank == 3

    def test_parameter_with_zero_row_and_column_is_flagged(self):
        # Parameter index 1 doesn't influence the state at all: its row
        # and column are zero, so e_1 is exactly a null-space direction.
        fim = np.diag([1.0, 0.0, 1.0])
        result = detect_redundant_parameters(fim)
        assert result.redundant_parameter_indices == [1]
        assert result.null_space_dimension == 1
        assert result.effective_rank == 2
        assert result.contributions[1] == [0]

    def test_coupled_null_direction_flags_both_parameters(self):
        # theta0 - theta1 direction is null: QFIM for a state depending
        # only on theta0 + theta1 (see test_qfim.redundant_state) has
        # eigenvector (1, -1)/sqrt(2) with eigenvalue 0.
        v = np.array([1.0, -1.0]) / np.sqrt(2)
        u = np.array([1.0, 1.0]) / np.sqrt(2)
        fim = 2.0 * np.outer(u, u) + 0.0 * np.outer(v, v)
        result = detect_redundant_parameters(fim)
        assert set(result.redundant_parameter_indices) == {0, 1}
        assert result.null_space_dimension == 1

    def test_fully_degenerate_qfim_flags_all_parameters(self):
        fim = np.zeros((3, 3))
        result = detect_redundant_parameters(fim)
        assert result.redundant_parameter_indices == [0, 1, 2]
        assert result.null_space_dimension == 3
        assert result.effective_rank == 0

    def test_invalid_contribution_threshold_raises(self):
        fim = np.eye(2)
        with pytest.raises(ValueError, match="contribution_threshold"):
            detect_redundant_parameters(fim, contribution_threshold=0.0)
        with pytest.raises(ValueError, match="contribution_threshold"):
            detect_redundant_parameters(fim, contribution_threshold=1.5)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            detect_redundant_parameters(np.zeros((2, 3)))
