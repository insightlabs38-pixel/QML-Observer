"""Framework adapters for qml_observer.

Converts framework-specific (or, for `GenericAdapter`, framework-neutral)
training information into calls against `QMLMonitor.update()`. PennyLane
and Qiskit adapters (Milestones 6 and 8, Volumes IX-X) will live here
alongside `GenericAdapter`.
"""

from qml_observer.adapters.generic import GenericAdapter

__all__ = ["GenericAdapter"]
