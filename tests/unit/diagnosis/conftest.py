"""Shared fixtures for diagnosis engine unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.schemas.gradient import summarize_gradient
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent


def make_observation(
    step: int,
    loss: float | None = None,
    gradient: np.ndarray | None = None,
    learning_rate: float | None = None,
    parameters=None,
) -> StepObservation:
    """Build a `StepObservation` with only the fields a test cares about."""
    optimizer = (
        OptimizerMetadata(name="test-optimizer", learning_rate=learning_rate)
        if learning_rate is not None
        else None
    )
    return StepObservation(
        training_event=TrainingEvent(run_id="test-run", step=step, loss=loss),
        gradient=summarize_gradient(gradient) if gradient is not None else None,
        optimizer=optimizer,
        parameters=parameters,
    )


@pytest.fixture
def run_state() -> RunState:
    return RunState(run_id="test-run", window_size=200)


@pytest.fixture
def obs_factory():
    return make_observation
