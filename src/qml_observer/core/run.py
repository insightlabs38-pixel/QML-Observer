"""Run ID generation and validation.

Milestone 2, Issue #14 ("Implement run IDs").

Run IDs are opaque, user-facing identifiers for a single monitored training
run (used in `TrainingEvent.run_id`, JSONL logs, reports, dashboards, and
future experiment-tracker integrations). Users may supply their own (e.g. to
match an existing experiment name); if omitted, `QMLMonitor` generates one
automatically so the zero-config path (`QMLMonitor()`) always works.
"""

from __future__ import annotations

import uuid

from qml_observer.schemas._validation import check_non_empty_str

#: Default prefix used by `generate_run_id` when none is supplied.
DEFAULT_RUN_ID_PREFIX = "run"


def generate_run_id(prefix: str = DEFAULT_RUN_ID_PREFIX) -> str:
    """Generate a reasonably unique, human-scannable run ID.

    Format: ``"{prefix}-{12 hex chars}"``, e.g. ``"run-3f9a2b7c1d4e"``.

    Uses `uuid.uuid4` truncated to 12 hex characters (48 bits of entropy),
    which is more than enough to make collisions negligible across a single
    machine's concurrently-started runs while staying short and readable in
    logs, JSONL output, and CLI summaries.

    Args:
        prefix: A short label prepended to the generated ID. Must be a
            non-empty string.

    Raises:
        TypeError: If `prefix` is not a string.
        ValueError: If `prefix` is empty/whitespace-only.
    """
    check_non_empty_str(prefix, "prefix")
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def validate_run_id(run_id: str) -> str:
    """Validate a user- or auto-supplied run_id, raising if malformed.

    Returns the run_id unchanged, so callers can write
    ``self.run_id = validate_run_id(run_id)``.

    Raises:
        TypeError: If `run_id` is not a string.
        ValueError: If `run_id` is empty/whitespace-only.
    """
    check_non_empty_str(run_id, "run_id")
    return run_id
