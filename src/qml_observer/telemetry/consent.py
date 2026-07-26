"""Opt-in telemetry consent (addendum §5).

Telemetry is **disabled by default**. Nothing is collected or sent until
the user explicitly opts in, either via the CLI (`qml-observer telemetry
enable`) or the Python API (`qml_observer.telemetry.enable()`). No dark
patterns: there is no opt-out-only design, and a non-interactive
environment (CI, piped input, no TTY) is never auto-enrolled -- it is
always treated as declined until a human explicitly opts in.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _default_consent_path() -> Path:
    override = os.environ.get("QML_OBSERVER_TELEMETRY_CONFIG")
    if override:
        return Path(override)
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "qml-observer" / "telemetry.json"


@dataclass
class ConsentState:
    enabled: bool = False


def load_consent(path: Path | None = None) -> ConsentState:
    """Load persisted consent state.

    Fails safe to disabled if no file exists yet, the file is malformed,
    or it can't be read for any reason -- consent must be explicit and
    positively recorded, never assumed.
    """
    resolved = path or _default_consent_path()
    try:
        with open(resolved) as f:
            data = json.load(f)
        return ConsentState(enabled=bool(data.get("enabled", False)))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return ConsentState(enabled=False)


def save_consent(state: ConsentState, path: Path | None = None) -> None:
    resolved = path or _default_consent_path()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w") as f:
        json.dump({"enabled": state.enabled}, f)


def is_enabled(path: Path | None = None) -> bool:
    """Whether telemetry is currently enabled. Defaults to `False`."""
    return load_consent(path).enabled


def enable(path: Path | None = None) -> None:
    """Explicitly opt in to anonymized telemetry."""
    save_consent(ConsentState(enabled=True), path)


def disable(path: Path | None = None) -> None:
    """Explicitly opt out of (or revoke) anonymized telemetry."""
    save_consent(ConsentState(enabled=False), path)


def has_been_asked(path: Path | None = None) -> bool:
    """Whether a consent decision has already been persisted."""
    resolved = path or _default_consent_path()
    return resolved.exists()


CONSENT_PROMPT = (
    "QML Observer can send anonymized, opt-in telemetry (detector "
    "name/version, anonymized threshold values, diagnosis outcome, "
    "framework name/version, a coarse qubit-count bucket, confidence "
    "score, and detection latency) to help improve default detector "
    "calibration community-wide.\n"
    "Never collected: raw gradients, loss values, circuit structure or "
    "ansatz source, parameter values, run IDs, file paths, or hostnames.\n"
    "See docs/development/telemetry.md for the full schema. This is "
    "entirely optional and off by default."
)


def prompt_for_consent(path: Path | None = None) -> bool:
    """Interactively prompt for telemetry consent, once, and persist the
    answer. Returns the resulting enabled state.

    Never prompts (and never enables) in a non-interactive environment
    (no TTY on stdin) -- e.g. CI, scripts with piped input, or
    non-interactive `python -c ...` invocations. In that case the
    decision is left unpersisted so an interactive user is still asked
    the next time they run something interactively.
    """
    if has_been_asked(path):
        return is_enabled(path)
    if not sys.stdin.isatty():
        return False
    print(CONSENT_PROMPT)
    answer = input("Enable anonymized telemetry? [y/N]: ").strip().lower()
    enabled = answer in ("y", "yes")
    save_consent(ConsentState(enabled=enabled), path)
    return enabled
