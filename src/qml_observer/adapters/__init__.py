"""Framework adapters for qml_observer.

Converts framework-specific (or, for `GenericAdapter`/`AutogradAdapter`,
framework-neutral) training information into calls against
`QMLMonitor.update()`. Alongside `GenericAdapter` and `AutogradAdapter`
(Milestone 14, Issue #100), `qml_observer.adapters.pennylane.PennyLaneAdapter`
(Milestone 6), `qml_observer.adapters.qiskit.QiskitAdapter` (Milestone 8),
`qml_observer.adapters.pytorch.PyTorchAdapter` (Milestone 14, Issue #98),
and `qml_observer.adapters.jax.JAXAdapter` (Milestone 14, Issue #99) live
in their own subpackages -- each gated behind its optional dependency
(`pip install qml-observer[pennylane]` / `[qiskit]` / `[torch]` / `[jax]`)
and, like `GenericAdapter`'s neighbors, not imported here so installs
without those extras are unaffected.

`AutogradAdapter` itself has no optional dependency (it duck-types tensor
conversion rather than importing any specific autodiff library), so it
*is* imported here alongside `GenericAdapter`.
"""

from qml_observer.adapters.autograd import AutogradAdapter
from qml_observer.adapters.generic import GenericAdapter

__all__ = ["GenericAdapter", "AutogradAdapter"]
