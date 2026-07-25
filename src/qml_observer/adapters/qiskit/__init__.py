"""Qiskit adapter subpackage (Milestone 8, Volume X).

Importing this subpackage requires the optional `qiskit` dependency
(`pip install qml-observer[qiskit]`). It is intentionally not imported by
`qml_observer.adapters` itself, so installs without Qiskit are unaffected
-- exactly the same convention as `qml_observer.adapters.pennylane`
(Milestone 6).
"""

from qml_observer.adapters.qiskit.adapter import QiskitAdapter

__all__ = ["QiskitAdapter"]
