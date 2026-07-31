"""Unit tests for qml_observer.advanced.geometry.loss_landscape."""

import math

import numpy as np
import pytest

from qml_observer.advanced.geometry.loss_landscape import (
    LandscapeSample,
    landscape_flatness,
    random_direction,
    sample_loss_landscape_1d,
    sample_loss_landscape_2d,
)


def bowl_loss(theta: np.ndarray) -> float:
    return float(np.sum(theta**2))


def flat_loss(theta: np.ndarray) -> float:
    return 0.5


class TestRandomDirection:
    def test_unit_norm(self):
        d = random_direction(5, seed=0)
        assert d.shape == (5,)
        assert np.linalg.norm(d) == pytest.approx(1.0)

    def test_seeded_is_reproducible(self):
        assert random_direction(4, seed=42) == pytest.approx(random_direction(4, seed=42))

    def test_non_positive_n_raises(self):
        with pytest.raises(ValueError, match="n_parameters"):
            random_direction(0)


class TestSampleLossLandscape1d:
    def test_basic_shape(self):
        sample = sample_loss_landscape_1d(bowl_loss, [0.0, 0.0], direction=[1.0, 0.0], n_points=11)
        assert isinstance(sample, LandscapeSample)
        assert sample.alphas.shape == (11,)
        assert sample.losses.shape == (11,)
        assert sample.betas is None

    def test_bowl_minimum_at_center(self):
        sample = sample_loss_landscape_1d(
            bowl_loss, [0.0, 0.0], direction=[1.0, 0.0], span=(-2.0, 2.0), n_points=41
        )
        center_idx = np.argmin(np.abs(sample.alphas))
        assert np.argmin(sample.losses) == center_idx

    def test_endpoints_included(self):
        sample = sample_loss_landscape_1d(
            bowl_loss, [0.0], direction=[1.0], span=(-3.0, 3.0), n_points=7
        )
        assert sample.alphas[0] == pytest.approx(-3.0)
        assert sample.alphas[-1] == pytest.approx(3.0)

    def test_mismatched_direction_shape_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            sample_loss_landscape_1d(bowl_loss, [0.0, 0.0], direction=[1.0])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="n_points"):
            sample_loss_landscape_1d(bowl_loss, [0.0], direction=[1.0], n_points=1)

    def test_invalid_span_raises(self):
        with pytest.raises(ValueError, match="span"):
            sample_loss_landscape_1d(bowl_loss, [0.0], direction=[1.0], span=(1.0, -1.0))


class TestSampleLossLandscape2d:
    def test_basic_shape(self):
        sample = sample_loss_landscape_2d(
            bowl_loss, [0.0, 0.0], direction1=[1.0, 0.0], direction2=[0.0, 1.0], n_points=5
        )
        assert sample.alphas.shape == (5,)
        assert sample.betas.shape == (5,)
        assert sample.losses.shape == (5, 5)

    def test_bowl_minimum_at_grid_center(self):
        sample = sample_loss_landscape_2d(
            bowl_loss,
            [0.0, 0.0],
            direction1=[1.0, 0.0],
            direction2=[0.0, 1.0],
            span=(-2.0, 2.0),
            n_points=9,
        )
        i, j = np.unravel_index(np.argmin(sample.losses), sample.losses.shape)
        assert sample.alphas[i] == pytest.approx(0.0, abs=1e-9)
        assert sample.betas[j] == pytest.approx(0.0, abs=1e-9)


class TestLandscapeFlatness:
    def test_flat_landscape_has_zero_range_and_std(self):
        sample = sample_loss_landscape_1d(flat_loss, [0.0], direction=[1.0], n_points=11)
        summary = landscape_flatness(sample)
        assert summary["range"] == pytest.approx(0.0)
        assert summary["std"] == pytest.approx(0.0)
        assert summary["mean"] == pytest.approx(0.5)

    def test_bowl_landscape_has_positive_range(self):
        sample = sample_loss_landscape_1d(
            bowl_loss, [0.0], direction=[1.0], span=(-1.0, 1.0), n_points=11
        )
        summary = landscape_flatness(sample)
        assert summary["range"] > 0.0
        assert summary["std"] > 0.0

    def test_nan_propagates(self):
        def nan_loss(theta):
            return float("nan")

        sample = sample_loss_landscape_1d(nan_loss, [0.0], direction=[1.0], n_points=5)
        summary = landscape_flatness(sample)
        assert math.isnan(summary["range"])
        assert math.isnan(summary["std"])
        assert math.isnan(summary["mean"])
