"""Milestone 14, Issue #99: basic JAX + QMLMonitor example.

A tiny pytree-parameterized model standing in for a hybrid
quantum-classical model expressed in JAX's typical functional style:
attach a `JAXAdapter`, compute `loss`/`gradients` with
`jax.value_and_grad`, and print the diagnosis each step. Mirrors
`examples/pennylane/basic_monitor.py`'s scope -- proving the plumbing
works end to end, not tuning detectors.

Run with:
    python examples/jax/basic_monitor.py

Requires: pip install qml-observer[jax]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qml_observer import QMLMonitor
from qml_observer.adapters.jax.adapter import JAXAdapter

N_STEPS = 30
LEARNING_RATE = 0.1


def init_params(key: jax.Array) -> dict:
    w_key, b_key = jax.random.split(key)
    return {"w": jax.random.normal(w_key, (4,)), "b": jax.random.normal(b_key, ())}


def loss_fn(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    prediction = jnp.dot(x, params["w"]) + params["b"]
    return jnp.mean((prediction - y) ** 2)


def sgd_update(params: dict, grads: dict) -> dict:
    return jax.tree_util.tree_map(lambda p, g: p - LEARNING_RATE * g, params, grads)


def main() -> None:
    key = jax.random.PRNGKey(0)
    params = init_params(key)
    x = jax.random.normal(jax.random.fold_in(key, 1), (16, 4))
    y = jax.random.normal(jax.random.fold_in(key, 2), (16,))

    # No detectors configured, same as the PennyLane basic_monitor example:
    # this script is about the integration plumbing, not detection.
    monitor = QMLMonitor(policy="log")
    adapter = JAXAdapter(monitor, params, optimizer_name="SGD", learning_rate=LEARNING_RATE)

    print(f"Run ID: {monitor.run_id}\n")
    for step in range(N_STEPS):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        diagnosis = adapter.record_step(step, loss, grads, params)
        print(f"step={step:>2}  loss={float(loss): .4f}  issue={diagnosis.issue.value}")
        params = sgd_update(params, grads)

    final = monitor.finish()
    print(f"\nFinal diagnosis: {final.issue.value} (confidence={final.confidence:.2f})")
    print(f"Total steps recorded: {monitor.state.step_count}")


if __name__ == "__main__":
    main()
