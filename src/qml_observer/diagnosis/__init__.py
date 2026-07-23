"""Diagnosis engine: combines detector outputs into a final diagnosis.

Milestone 4 (Volume VII). `qml_observer.diagnosis.engine.DiagnosisEngine`
drives detectors each step; `qml_observer.diagnosis.scoring` (Issue #30)
holds the standalone confidence-combination primitive it delegates to;
`explanations.py` (Issue #31) renders a `DiagnosisResult` as human-readable
text.
"""

from qml_observer.diagnosis.engine import DiagnosisEngine
from qml_observer.diagnosis.explanations import explain
from qml_observer.diagnosis.scoring import combine_detector_results

__all__ = ["DiagnosisEngine", "combine_detector_results", "explain"]
