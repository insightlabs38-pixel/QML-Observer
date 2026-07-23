"""Diagnosis engine: combines detector outputs into a final diagnosis.

Milestone 4 (Volume VII). See `qml_observer.diagnosis.engine` for the
`DiagnosisEngine` itself; `scoring.py` (Issue #30) and `explanations.py`
(Issue #31) will extend this package with a standalone, reusable scoring
primitive and richer human-readable explanations.
"""

from qml_observer.diagnosis.engine import DiagnosisEngine

__all__ = ["DiagnosisEngine"]
