"""Convenience launcher for the dashboard app (Issue #76; hardened per
Issue #82b).

Binds to `127.0.0.1` by default. The dashboard has no authentication, so
serving it on a non-loopback interface exposes run data (and, per
`docs/development/data_handling.md`, potentially proprietary circuit/loss
data) to anything else able to reach that interface.

Per Issue #82b, this is a *refuse-by-default* safeguard, not just a
warning: passing a non-loopback `host` raises `ValueError` unless the
caller also passes `allow_non_loopback=True`, at which point a warning is
printed to stderr and the server starts. This mirrors the Milestone 10
webhook safeguard's `allow_internal_targets` opt-out pattern
(`qml_observer.integrations.security`) -- see `SECURITY.md` for the
documented security-boundary writeup this issue also called for.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qml_observer.dashboard.data_source import DashboardDataSource

#: Hosts considered "local" enough to need no extra flag/warning.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def run_dashboard(
    source: DashboardDataSource,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    history_dir: str | Path | None = None,
    allow_non_loopback: bool = False,
    log_level: str = "warning",
) -> None:
    """Serve the dashboard app (blocking) via `uvicorn`.

    Args:
        source: The `DashboardDataSource` to read run data from -- see
            `dashboard/data_source.py`.
        host: Interface to bind. Defaults to loopback-only
            (`127.0.0.1`). Binding anywhere else requires
            `allow_non_loopback=True` (see below) -- this is a research
            tool's dashboard, not a hardened multi-user service.
        port: TCP port to bind. Defaults to `8765`.
        history_dir: Optional directory of other finalized runs' JSONL
            logs to expose via the run-history panel (Issue #81). See
            `dashboard.app.create_app`.
        allow_non_loopback: Must be `True` to bind `host` to anything
            other than loopback. Defaults to `False` (refuse), matching
            the project's existing "refuse unless explicitly allowed"
            posture for other network-facing surfaces (the webhook SSRF
            safeguard). When `True`, a warning is still printed to stderr
            every time, since this is a meaningfully riskier
            configuration each time it happens, not just the first.
        log_level: `uvicorn` log level. Defaults to `"warning"` to keep
            routine polling requests out of the terminal by default.

    Raises:
        ValueError: If `host` is not a loopback address and
            `allow_non_loopback` is not `True`.
        ImportError: If the `dashboard` extra (`fastapi`/`uvicorn`) is not
            installed. See `dashboard/app.py`'s module docstring.
    """
    if host not in _LOOPBACK_HOSTS and not allow_non_loopback:
        raise ValueError(
            f"refusing to bind the qml-observer dashboard to non-loopback host {host!r}: "
            "the dashboard has no authentication, so anything able to reach this "
            "host/port could read this run's data (potentially including proprietary "
            "circuit/loss data -- see docs/development/data_handling.md). Pass "
            "allow_non_loopback=True if you've deliberately decided this is safe on "
            "your network."
        )
    if host not in _LOOPBACK_HOSTS:
        print(
            f"warning: qml-observer dashboard binding to non-loopback host {host!r}. "
            "The dashboard has no authentication -- anything able to reach this "
            "host/port can read this run's data. Only do this on a trusted network.",
            file=sys.stderr,
        )

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised via import-error test
        raise ImportError(
            "The QML Observer dashboard requires the optional 'dashboard' extra. "
            "Install it with: pip install 'qml-observer[dashboard]'"
        ) from exc

    from qml_observer.dashboard.app import create_app

    app = create_app(source, history_dir=history_dir)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
