"""Framework-agnostic data schemas for qml_observer.

This package holds the internal event/data model that adapters convert
framework-specific information into. Schemas are added incrementally
(see Milestone 1, Issues #4-#10 in the project blueprint).
"""

from qml_observer.schemas.circuit import CircuitMetadata
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType
from qml_observer.schemas.gradient import GradientSnapshot, summarize_gradient
from qml_observer.schemas.optimizer import OptimizerMetadata
from qml_observer.schemas.training import TrainingEvent

__all__ = [
    "TrainingEvent",
    "GradientSnapshot",
    "summarize_gradient",
    "CircuitMetadata",
    "OptimizerMetadata",
    "IssueType",
    "DiagnosisResult",
]
