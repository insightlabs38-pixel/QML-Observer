"""Detector layer: framework-agnostic training-pathology detectors.

Milestone 4 (Volume V-VI). Every detector implements `BaseDetector` and
produces raw, per-detector `DetectorResult`s; the `DiagnosisEngine`
(`qml_observer.diagnosis`) combines them into a single, explainable
`DiagnosisResult`. Detectors never decide the final diagnosis themselves
-- see blueprint Volume VII for why that separation matters.

Milestone 14, Issue #103 adds `qml_observer.detectors.plugins` for
discovering third-party detectors registered via the
`qml_observer.detectors` entry-point group; its most commonly used
functions are re-exported here alongside the built-in detectors.
"""

from qml_observer.detectors.barren_plateau import BarrenPlateauDetector
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.detectors.convergence import ConvergenceDetector
from qml_observer.detectors.noise import NoiseDetector
from qml_observer.detectors.plugins import (
    DetectorPluginError,
    discover_detector_plugins,
    list_detector_plugins,
    load_detector_plugins,
)
from qml_observer.detectors.stagnation import StagnationDetector

__all__ = [
    "BaseDetector",
    "DetectorResult",
    "BarrenPlateauDetector",
    "StagnationDetector",
    "ConvergenceDetector",
    "NoiseDetector",
    "DetectorPluginError",
    "discover_detector_plugins",
    "list_detector_plugins",
    "load_detector_plugins",
]
