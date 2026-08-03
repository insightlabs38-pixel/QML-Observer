"""Integration tests for the JAX adapter (Milestone 14, Issue #99).

Unlike `tests/unit/adapters/test_jax.py`, which exercises `JAXAdapter` in
isolation, these tests drive a full, real training loop:
`jax.value_and_grad`-computed gradients over a real pytree of parameters,
the real detector stack, and the real `ActionPolicy` -- end to end,
exactly as `examples/jax/` does. Skipped entirely if the optional `jax`
dependency isn't installed.
"""

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from qml_observer import QMLMonitor  # noqa: E402
from qml_observer.adapters.jax.adapter import JAXAdapter  # noqa: E402
from qml_observer.detectors.barren_plateau import BarrenPlateauDetector  # noqa: E402
from qml_observer.detectors.convergence import ConvergenceDetector  # noqa: E402
from qml_observer.detectors.stagnation import StagnationDetector  # noqa: E402
from qml_observer.schemas.diagnosis import IssueType  # noqa: E402

PATIENCE = 15
LEARNING_RATE = 0.1


def _detectors():
    return [
        BarrenPlateauDetector(patience=PATIENCE),
        StagnationDetector(patience=PATIENCE),
        ConvergenceDetector(patience=PATIENCE, loss_threshold=1e-2),
    ]


def _healthy_loss(params, x, y):
    prediction = jnp.dot(x, params["w"]) + params["b"]
    return jnp.mean((prediction - y) ** 2)


def _frozen_loss(params, x, y):
    """A loss whose gradient w.r.t. every leaf is always exactly zero.

    `params["dead"]` never enters the computation (multiplied by zero),
    and the "prediction" is a constant well above `y`'s scale, so the
    loss stays flat and non-trivially large -- distinct from a model that
    collapses to a genuinely converged (loss == 0) constant, which the
    diagnosis engine correctly reports as convergence rather than a
    plateau.
    """
    del x
    constant_prediction = 3.0 + 0.0 * jnp.sum(params["dead"])
    return jnp.mean((constant_prediction - y) ** 2)


def _sgd_update(params, grads):
    return jax.tree_util.tree_map(lambda p, g: p - LEARNING_RATE * g, params, grads)


class TestHealthyConvergence:
    def test_does_not_falsely_stop_on_real_convergence(self):
        key = jax.random.PRNGKey(0)
        params = {"w": jax.random.normal(key, (4,)), "b": jnp.array(0.0)}
        x = jax.random.normal(jax.random.fold_in(key, 1), (16, 4))
        y = jax.random.normal(jax.random.fold_in(key, 2), (16,)) * 0.1

        monitor = QMLMonitor(detectors=_detectors(), policy="stop")
        adapter = JAXAdapter(monitor, params, optimizer_name="SGD", learning_rate=LEARNING_RATE)

        stopped_early = False
        for step in range(150):
            loss, grads = jax.value_and_grad(_healthy_loss)(params, x, y)
            adapter.record_step(step, loss, grads, params)
            params = _sgd_update(params, grads)
            if monitor.should_stop():
                stopped_early = True
                break

        final = monitor.finish()
        assert stopped_early is False
        assert final.issue in (
            IssueType.HEALTHY,
            IssueType.CONVERGED,
            IssueType.INSUFFICIENT_EVIDENCE,
        )


class TestGradientCollapse:
    def test_detects_sustained_gradient_collapse(self):
        key = jax.random.PRNGKey(0)
        params = {"dead": jnp.zeros(6)}
        x = jax.random.normal(key, (16, 4))
        y = jnp.zeros(16)

        monitor = QMLMonitor(detectors=_detectors(), policy="stop")
        adapter = JAXAdapter(monitor, params, optimizer_name="SGD", learning_rate=LEARNING_RATE)

        stopped_early = False
        for step in range(120):
            loss, grads = jax.value_and_grad(_frozen_loss)(params, x, y)
            adapter.record_step(step, loss, grads, params)
            params = _sgd_update(params, grads)
            if monitor.should_stop():
                stopped_early = True
                break

        final = monitor.finish()
        assert stopped_early is True
        assert final.issue == IssueType.POSSIBLE_BARREN_PLATEAU
