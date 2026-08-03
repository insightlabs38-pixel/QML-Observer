"""PyTorch adapter subpackage (Milestone 14, Issue #98).

Importing this subpackage requires the optional `torch` dependency
(`pip install qml-observer[torch]`). It is intentionally not imported by
`qml_observer.adapters` itself, so installs without PyTorch are
unaffected -- the same convention as `qml_observer.adapters.pennylane`
and `qml_observer.adapters.qiskit`.
"""

from qml_observer.adapters.pytorch.adapter import PyTorchAdapter

__all__ = ["PyTorchAdapter"]
