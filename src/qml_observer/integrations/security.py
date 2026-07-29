"""Minimal SSRF safeguard for user-supplied webhook URLs.

Milestone 10, Issue #75c ("Threat-model webhook URLs").

A `WebhookAction` URL is an outbound HTTP call to an arbitrary,
user-supplied address. That's an ordinary, low-risk thing for a
single-user script (you're only configuring where *your own* alerts go).
It stops being low-risk the moment `QMLMonitor`/`WebhookAction` is ever
driven by input Python doesn't control -- e.g. a shared/multi-tenant
service that accepts a "webhook URL" from a caller and constructs a
`WebhookAction` with it. In that shape, an attacker-supplied URL pointing
at `http://169.254.169.254/...` (cloud metadata endpoints), `localhost`,
or an internal service is a classic SSRF vector.

This module implements the "minimal safeguard" the addendum's governance
posture (§10, extended here to network-facing config) calls for: refuse
*obviously* internal-looking targets by default, and require an explicit
`allow_internal_targets=True` opt-in to use one anyway (e.g. for local
development against the `examples/generic/webhook_alerting.py`-style local
receiver).

**Explicitly out of scope / residual risk** (do not oversell this check):

- This is a **string/literal-IP check only** -- it does not resolve
  hostnames via DNS. A public-looking hostname that a DNS response (or
  DNS rebinding attack) later resolves to an internal address is *not*
  caught here. A full defense would re-check the resolved IP at request
  time (or route the request through an egress proxy that enforces this),
  which this minimal v1 safeguard does not do.
- It does not inspect HTTP redirects: if the configured URL's server
  responds with a redirect to an internal address, `urllib.request`
  follows it and this module does not re-validate the redirect target.

Document this boundary explicitly (per the project's existing "state the
security boundary, don't silently assume it away" posture -- see
`SECURITY.md`) rather than presenting this as complete SSRF protection.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

#: Hostnames that are unambiguously "this machine" regardless of DNS.
_INTERNAL_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}

#: Schemes a webhook is allowed to target. Anything else (file://, ftp://,
#: gopher://, etc.) is refused outright -- a webhook is an HTTP(S)
#: notification channel, not a general-purpose URL fetcher.
_ALLOWED_SCHEMES = {"http", "https"}


class UnsafeWebhookURLError(ValueError):
    """Raised when a webhook URL is refused for looking obviously internal
    or using a disallowed scheme, and the caller has not explicitly opted
    in via `allow_internal_targets=True`.
    """


def _is_obviously_internal_host(hostname: str) -> bool:
    """Best-effort, DNS-free check for a loopback/private/link-local host.

    Returns `True` for `localhost`-style names and for any hostname that
    is *itself* a literal IP address in a loopback, link-local, private,
    reserved, or unspecified range. Returns `False` for anything else,
    including ordinary public domain names -- which this function cannot
    (and does not attempt to) resolve.
    """
    if not hostname:
        return True  # no host at all is not a usable public target
    lowered = hostname.strip("[]").lower()  # strip IPv6 literal brackets
    if lowered in _INTERNAL_HOSTNAMES or lowered.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        return False  # a hostname, not a literal IP -- can't judge without DNS
    return (
        ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_reserved or ip.is_unspecified
    )


def check_webhook_url(url: str, *, allow_internal_targets: bool) -> None:
    """Validate `url` is an http(s) URL and, unless explicitly allowed,
    does not obviously target an internal/loopback/link-local address.

    Raises:
        UnsafeWebhookURLError: If the scheme isn't `http`/`https`, or the
            host looks internal and `allow_internal_targets` is `False`.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise UnsafeWebhookURLError(
            f"Refusing webhook URL with scheme {parsed.scheme!r}: only "
            f"{sorted(_ALLOWED_SCHEMES)} are supported."
        )
    hostname = parsed.hostname or ""
    if _is_obviously_internal_host(hostname) and not allow_internal_targets:
        raise UnsafeWebhookURLError(
            f"Refusing to configure a webhook targeting an internal-looking "
            f"host ({hostname!r}). This guards against SSRF if a webhook URL "
            f"is ever accepted from an untrusted caller (e.g. a shared/"
            f"multi-tenant deployment). Pass allow_internal_targets=True if "
            f"this is intentional (e.g. local development/testing against a "
            f"local receiver)."
        )
