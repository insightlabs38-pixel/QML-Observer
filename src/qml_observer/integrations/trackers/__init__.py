"""Experiment-tracker integrations (Milestone 14, Issue #101).

Writes `QMLMonitor` output into *existing* experiment-tracking
infrastructure (MLflow, Weights & Biases) rather than adding a competing
tracker of qml_observer's own (plan.md §25 / future_milestones_plan.md
Milestone 14: "optional adapters writing `QMLMonitor` output into existing
tracking infra"). Both trackers implement the same `RunReporter` duck
type `QMLMonitor(reporter=...)` already knows how to drive
(`record_event`/`record_diagnosis`/`finalize`, see `reporting/reporter.py`),
so either can be passed directly in place of (or alongside -- see
`BaseExperimentTracker`'s docstring) `RunReporter`.

Like `qml_observer.adapters.pennylane`/`.qiskit`/`.pytorch`/`.jax`, the two
concrete trackers each depend on one optional third-party package and are
therefore *not* imported here -- only `base.py`'s shared, dependency-free
plumbing is:

    from qml_observer.integrations.trackers.mlflow_tracker import MLflowTracker
    from qml_observer.integrations.trackers.wandb_tracker import WandbTracker
"""

from qml_observer.integrations.trackers.base import (
    BaseExperimentTracker,
    diagnosis_metrics,
    event_metrics,
)

__all__ = ["BaseExperimentTracker", "event_metrics", "diagnosis_metrics"]
