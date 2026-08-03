"""JAX adapter subpackage (Milestone 14, Issue #99).

Importing this subpackage requires the optional `jax` dependency
(`pip install qml-observer[jax]`). It is intentionally not imported by
`qml_observer.adapters` itself, so installs without JAX are unaffected --
the same convention as `qml_observer.adapters.pennylane`,
`qml_observer.adapters.qiskit`, and `qml_observer.adapters.pytorch`.
"""

from qml_observer.adapters.jax.adapter import JAXAdapter

__all__ = ["JAXAdapter"]
