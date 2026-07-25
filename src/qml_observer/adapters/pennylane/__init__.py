"""PennyLane adapter subpackage (Milestone 6, Volume IX).

Importing this subpackage requires the optional `pennylane` dependency
(`pip install qml-observer[pennylane]`). It is intentionally not imported
by `qml_observer.adapters` itself, so installs without PennyLane are
unaffected.
"""

from qml_observer.adapters.pennylane.adapter import PennyLaneAdapter

__all__ = ["PennyLaneAdapter"]
