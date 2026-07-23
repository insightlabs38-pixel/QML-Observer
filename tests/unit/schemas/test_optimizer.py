"""Unit tests for qml_observer.schemas.optimizer.OptimizerMetadata."""

import pytest

from qml_observer.schemas.optimizer import OptimizerMetadata


class TestConstruction:
    def test_name_only(self):
        opt = OptimizerMetadata(name="Adam")
        assert opt.name == "Adam"
        assert opt.learning_rate is None
        assert opt.gradient_method is None

    def test_full_construction(self):
        opt = OptimizerMetadata(
            name="SPSA", learning_rate=0.05, gradient_method="spsa-approximation"
        )
        assert opt.learning_rate == 0.05
        assert opt.gradient_method == "spsa-approximation"

    def test_name_is_required(self):
        with pytest.raises(TypeError):
            OptimizerMetadata()  # type: ignore[call-arg]

    def test_zero_learning_rate_is_allowed(self):
        """Models an effectively-frozen optimizer (StagnationDetector)."""
        opt = OptimizerMetadata(name="Frozen", learning_rate=0.0)
        assert opt.learning_rate == 0.0


class TestValidation:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            OptimizerMetadata(name="")

    def test_negative_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            OptimizerMetadata(name="Adam", learning_rate=-0.1)

    def test_nan_learning_rate_raises(self):
        """Unlike loss/gradients, learning_rate is config -- NaN/Inf here
        indicates an adapter bug, not a training signal, and is rejected."""
        with pytest.raises(ValueError, match="finite"):
            OptimizerMetadata(name="Adam", learning_rate=float("nan"))

    def test_inf_learning_rate_raises(self):
        with pytest.raises(ValueError, match="finite"):
            OptimizerMetadata(name="Adam", learning_rate=float("inf"))

    def test_empty_gradient_method_raises(self):
        with pytest.raises(ValueError, match="gradient_method"):
            OptimizerMetadata(name="Adam", gradient_method="")
