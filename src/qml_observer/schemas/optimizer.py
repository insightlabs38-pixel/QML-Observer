"""OptimizerMetadata schema.

Metadata describing the classical optimizer driving a variational
training loop. This is deliberately lightweight — it exists so
detectors/diagnosis can contextualize gradient and loss behavior (e.g.
"gradient collapse under Adam with lr=0.1" reads differently than under
a natural-gradient optimizer), not to model optimizer internals or state.
"""

from dataclasses import dataclass

from qml_observer.schemas._validation import check_finite_number, check_non_empty_str


@dataclass
class OptimizerMetadata:
    """Metadata describing the optimizer used for a training run.

    Attributes:
        name: Optimizer name (e.g. "Adam", "GradientDescent", "SPSA",
            "QNSPSA"). Required, since detectors/reporting reference the
            optimizer by name even when no other metadata is available.
        learning_rate: Learning rate / step size, if applicable and known.
        gradient_method: Name of the gradient computation method in use
            (e.g. "parameter-shift", "adjoint", "finite-difference",
            "spsa-approximation"). Kept here as well as on
            `GradientSnapshot.method` since the optimizer configuration is
            often where this is actually specified upstream.
    """

    name: str
    learning_rate: float | None = None
    gradient_method: str | None = None

    def __post_init__(self) -> None:
        check_non_empty_str(self.name, "name")
        # learning_rate is configuration, not an observed training signal
        # (unlike loss/gradients), so unlike those fields NaN/Inf here
        # indicates a genuine adapter/config bug and should be rejected.
        # 0 is allowed deliberately: it models an "effectively frozen"
        # optimizer (see StagnationDetector, blueprint Volume VI-2).
        check_finite_number(self.learning_rate, "learning_rate")
        if self.learning_rate is not None and self.learning_rate < 0:
            raise ValueError(f"learning_rate must be >= 0, got {self.learning_rate}")
        if self.gradient_method is not None:
            check_non_empty_str(self.gradient_method, "gradient_method")
