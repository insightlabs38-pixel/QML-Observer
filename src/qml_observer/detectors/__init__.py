"""Detector layer: framework-agnostic training-pathology detectors.

Milestone 4 (Volume V-VI). Every detector implements `BaseDetector` and
produces raw, per-detector `DetectorResult`s; the `DiagnosisEngine`
(`qml_observer.diagnosis`) combines them into a single, explainable
`DiagnosisResult`. Detectors never decide the final diagnosis themselves
-- see blueprint Volume VII for why that separation matters.
"""

from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.detectors.convergence import ConvergenceDetector
from qml_observer.detectors.stagnation import StagnationDetector

__all__ = [
    "BaseDetector",
    "DetectorResult",
    "BarrenPlateauDetector",
    "StagnationDetector",
    "ConvergenceDetector",
]
