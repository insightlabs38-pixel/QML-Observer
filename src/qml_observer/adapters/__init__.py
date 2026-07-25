"""Framework adapters for qml_observer.

Converts framework-specific (or, for `GenericAdapter`, framework-neutral)
training information into calls against `QMLMonitor.update()`. Alongside
`GenericAdapter`, `qml_observer.adapters.pennylane.PennyLaneAdapter`
(Milestone 6) and `qml_observer.adapters.qiskit.QiskitAdapter` (Milestone
8) live in their own subpackages -- each gated behind its optional
dependency (`pip install qml-observer[pennylane]` /
`qml-observer[qiskit]`) and, like `GenericAdapter`'s neighbors, not
imported here so installs without those extras are unaffected.
"""

from qml_observer.adapters.generic import GenericAdapter

__all__ = ["GenericAdapter"]
