"""CircuitMetadata schema.

Circuit-level metadata that influences diagnosis (e.g. distinguishing a
deep, highly-entangling ansatz that is expected to be hard to train from
a shallow one that shouldn't be plateauing). Populated by adapters from
whatever circuit introspection the underlying framework exposes; every
field is optional because not all frameworks/execution paths expose all
of this information.

Scope note: this intentionally matches the blueprint's Volume II spec
(7 fields) rather than the full superset of circuit properties described
narratively in plan.md (e.g. hardware connectivity profile, noise model).
Those are backend/noise-diagnostic concerns that belong with the noise
detector work (Milestone 9) and are explicitly out of scope for real
hardware until funding is secured (see addendum §4); adding placeholder
fields for them here now would be speculative. This schema can grow
field-by-field, non-breaking, when that work lands.
"""

from dataclasses import dataclass

from qml_observer.schemas._validation import check_non_empty_str, check_non_negative_int


@dataclass
class CircuitMetadata:
    """Metadata describing the parameterized quantum circuit being trained.

    Attributes:
        n_qubits: Number of qubits in the circuit.
        depth: Circuit depth (longest path of sequential gates).
        n_parameters: Number of trainable parameters.
        n_gates: Total gate count.
        n_entangling_gates: Count of entangling (multi-qubit) gates.
        ansatz_name: Name/identifier of the ansatz (e.g. "StronglyEntanglingLayers").
        initialization: Name of the parameter initialization strategy used
            (e.g. "random_uniform", "zeros", "reduced_domain").
    """

    n_qubits: int | None = None
    depth: int | None = None
    n_parameters: int | None = None
    n_gates: int | None = None
    n_entangling_gates: int | None = None
    ansatz_name: str | None = None
    initialization: str | None = None

    def __post_init__(self) -> None:
        check_non_negative_int(self.n_qubits, "n_qubits")
        check_non_negative_int(self.depth, "depth")
        check_non_negative_int(self.n_parameters, "n_parameters")
        check_non_negative_int(self.n_gates, "n_gates")
        check_non_negative_int(self.n_entangling_gates, "n_entangling_gates")
        if self.n_gates is not None and self.n_entangling_gates is not None:
            if self.n_entangling_gates > self.n_gates:
                raise ValueError(
                    f"n_entangling_gates ({self.n_entangling_gates}) cannot exceed "
                    f"n_gates ({self.n_gates})"
                )
        if self.ansatz_name is not None:
            check_non_empty_str(self.ansatz_name, "ansatz_name")
        if self.initialization is not None:
            check_non_empty_str(self.initialization, "initialization")
