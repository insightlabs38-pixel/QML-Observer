# Event model

`qml_observer.schemas` defines the framework-agnostic data every other
layer consumes. Adapters translate framework objects into these; nothing
downstream ever imports PennyLane or Qiskit types directly.

- **`TrainingEvent`** (`schemas/training.py`) -- one observed step:
  `run_id`, `step`, `loss`, `epoch`, `timestamp`, `wall_time`.
- **`GradientSnapshot`** (`schemas/gradient.py`) -- a structured gradient
  summary: `norm_l2`, `mean_abs`, `variance`, `min_value`/`max_value`,
  `median_abs`, optional `snr`/`uncertainty`/`method`. Built via
  `summarize_gradient()` so detectors never touch raw gradient arrays
  directly (keeping memory use bounded -- addendum's performance rules).
- **`CircuitMetadata`** (`schemas/circuit.py`) -- qubit count, depth,
  parameter/gate/entangling-gate counts, ansatz name, initialization
  strategy.
- **`OptimizerMetadata`** (`schemas/optimizer.py`) -- optimizer name,
  learning rate, gradient computation method.
- **`DiagnosisResult`** (`schemas/diagnosis.py`) -- the diagnosis engine's
  final verdict: `issue` (an `IssueType`), `confidence`, `severity`,
  `evidence`, `recommendations`, and the addendum §1 `degraded`/
  `degraded_reason` pair.

All five are validated on construction (`schemas/_validation.py`) --
malformed values raise immediately rather than propagating silently into a
detector's arithmetic.
