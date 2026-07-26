"""Telemetry payload schema (addendum §5).

Every telemetry record is built exclusively through `build_telemetry_record()`
below, and only ever contains the anonymized fields listed in
`TelemetryRecord`. It intentionally excludes raw gradients, loss values,
circuit structure/ansatz source, parameter values, run IDs, file paths,
and hostnames -- see `docs/development/telemetry.md` for the full,
audit-ready schema documentation and `docs/development/data_handling.md`
for the surrounding privacy policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = "1"

# (inclusive upper bound, label) -- coarse buckets so no exact qubit count
# is ever transmitted.
_QUBIT_BUCKETS: list[tuple[int, str]] = [
    (5, "1-5"),
    (10, "6-10"),
    (20, "11-20"),
    (50, "21-50"),
    (100, "51-100"),
]

# Attribute-name fragments that identify a detector's numeric threshold
# constructor arguments, generically, without hardcoding per-detector
# knowledge here.
_THRESHOLD_KEY_MARKERS = ("threshold", "patience")


def bucket_qubit_count(n_qubits: int | None) -> str | None:
    """Bucket an exact qubit count into a coarse range.

    Returns `None` if `n_qubits` is `None`. Never returns the exact count.
    """
    if n_qubits is None:
        return None
    for ceiling, label in _QUBIT_BUCKETS:
        if n_qubits <= ceiling:
            return label
    return "101+"


def extract_detector_thresholds(detector: object) -> dict[str, float]:
    """Best-effort, generic extraction of a detector's numeric threshold
    and patience settings, for anonymized calibration telemetry only.

    Reads only public numeric attributes whose name contains "threshold"
    or equals/contains "patience" (e.g. `gradient_threshold`,
    `variance_threshold`, `patience`). Never reads gradient/loss data,
    since detectors do not store raw training data as attributes.
    """
    result: dict[str, float] = {}
    for key, value in vars(detector).items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if any(marker in key for marker in _THRESHOLD_KEY_MARKERS):
            clean_key = key.lstrip("_")
            result[f"{type(detector).__name__}.{clean_key}"] = float(value)
    return result


@dataclass(frozen=True)
class TelemetryRecord:
    """A single anonymized diagnosis summary -- the only shape of data
    telemetry ever sends. See the module docstring for what is excluded.
    """

    schema_version: str
    package_version: str
    detector_names: list[str]
    thresholds: dict[str, float]
    issue: str
    confidence: float
    framework: str | None
    qubit_bucket: str | None
    detection_latency_steps: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_telemetry_record(
    *,
    detector_names: list[str],
    thresholds: dict[str, float],
    issue: str,
    confidence: float,
    framework: str | None = None,
    n_qubits: int | None = None,
    detection_latency_steps: int | None = None,
) -> TelemetryRecord:
    """Build a `TelemetryRecord` from already-anonymized inputs.

    Callers are responsible for ensuring `thresholds` and `detector_names`
    do not themselves carry identifying information (they normally come
    from `extract_detector_thresholds()` and detector class names, which
    is safe by construction).
    """
    from qml_observer import __version__ as package_version

    return TelemetryRecord(
        schema_version=SCHEMA_VERSION,
        package_version=package_version,
        detector_names=sorted(detector_names),
        thresholds=dict(thresholds),
        issue=issue,
        confidence=confidence,
        framework=framework,
        qubit_bucket=bucket_qubit_count(n_qubits),
        detection_latency_steps=detection_latency_steps,
    )
