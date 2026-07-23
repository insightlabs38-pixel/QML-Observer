"""Framework-agnostic monitoring core for qml_observer.

Milestone 2 (Volume III): the `QMLMonitor` public API, its rolling run
state, run-ID handling, and internal step-event structures. Detectors,
statistics, and the diagnosis engine (Milestones 3-4) are consumed here via
a documented seam (`QMLMonitor._evaluate`) but do not exist yet.
"""

from qml_observer.core.events import StepObservation
from qml_observer.core.monitor import QMLMonitor
from qml_observer.core.run import generate_run_id, validate_run_id
from qml_observer.core.state import RunState

__all__ = [
    "QMLMonitor",
    "StepObservation",
    "RunState",
    "generate_run_id",
    "validate_run_id",
]
