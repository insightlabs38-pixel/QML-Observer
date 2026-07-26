"""Opt-in, anonymized telemetry for community-wide detector calibration.

**Disabled by default.** See `docs/development/telemetry.md` for the full
schema and `docs/development/data_handling.md` for the surrounding
privacy/data-retention policy (addendum §5).

Quick start:

    >>> from qml_observer.telemetry import enable, TelemetryCollector
    >>> enable()  # explicit opt-in, persisted to ~/.config/qml-observer/
    >>> collector = TelemetryCollector()  # or CLI: `qml-observer telemetry enable`
    >>> monitor = QMLMonitor(telemetry_collector=collector, telemetry_framework="pennylane")
"""

from qml_observer.telemetry.collector import TelemetryCollector
from qml_observer.telemetry.consent import (
    disable,
    enable,
    is_enabled,
    prompt_for_consent,
)
from qml_observer.telemetry.schema import (
    TelemetryRecord,
    build_telemetry_record,
    extract_detector_thresholds,
)

__all__ = [
    "TelemetryCollector",
    "TelemetryRecord",
    "build_telemetry_record",
    "disable",
    "enable",
    "extract_detector_thresholds",
    "is_enabled",
    "prompt_for_consent",
]
