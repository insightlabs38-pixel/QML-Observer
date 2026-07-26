"""Anonymized telemetry collector (addendum §5).

`TelemetryCollector` is a no-op unless the user has separately enabled
telemetry via `qml_observer.telemetry.enable()` or `qml-observer telemetry
enable` (see `consent.py`). This release ships no bundled telemetry
backend: if no endpoint is configured, enabled records are queued locally
as JSON Lines instead of being transmitted anywhere, so the collector is
fully functional and testable without making any unannounced network
call. Set `QML_OBSERVER_TELEMETRY_ENDPOINT` (or pass `endpoint=`) to
actually transmit records to a configured collection service.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

from qml_observer.telemetry.consent import is_enabled
from qml_observer.telemetry.schema import TelemetryRecord

logger = logging.getLogger("qml_observer.telemetry")


def _default_queue_path() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "qml-observer" / "telemetry_queue.jsonl"


class TelemetryCollector:
    """Collects, and optionally transmits, anonymized diagnosis summaries.

    Always a no-op if telemetry is not enabled (`consent.is_enabled()`),
    regardless of whether an endpoint is configured -- opt-in is checked
    on every call, not just at construction time, so revoking consent
    mid-process takes effect immediately.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        queue_path: Path | str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._endpoint = endpoint or os.environ.get("QML_OBSERVER_TELEMETRY_ENDPOINT")
        self._queue_path = Path(queue_path) if queue_path is not None else None
        self._timeout = timeout

    def maybe_collect(self, record: TelemetryRecord) -> bool:
        """Transmit (or locally queue) `record` iff telemetry is enabled.

        Returns whether the record was collected. Never raises: any
        failure is logged at `warning` level and swallowed, matching the
        project's fail-open policy (addendum §1) -- telemetry must never
        be able to disrupt a training run.
        """
        if not is_enabled():
            return False
        try:
            if self._endpoint:
                self._send(record)
            else:
                self._queue_locally(record)
            return True
        except Exception:
            logger.warning(
                "qml_observer telemetry: submission failed; continuing without it.",
                exc_info=True,
            )
            return False

    def _send(self, record: TelemetryRecord) -> None:
        assert self._endpoint is not None
        payload = json.dumps(record.to_dict()).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout):  # noqa: S310
            pass

    def _queue_locally(self, record: TelemetryRecord) -> None:
        path = self._queue_path or _default_queue_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
