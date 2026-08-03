# Security Policy

## Supported versions

QML Observer is in public beta (`0.x`). Security fixes are made against
the latest released `0.x` version; older `0.x` releases are not
separately maintained.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security
vulnerabilities. Instead, use GitHub's private vulnerability reporting
("Report a vulnerability" under the Security tab of this repository) or
open a private security advisory. Include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce, or a minimal proof-of-concept.
- The affected version(s).

We aim to acknowledge reports within a few business days.

## Scope notes

- QML Observer is a local, non-invasive monitoring library. By default it
  writes nothing off-machine; see `docs/development/data_handling.md` for
  the current data-handling model.
- A third-party detector plugin API (Milestone 14, Issue #103,
  `qml_observer.detectors.plugins`) discovers and imports detectors
  registered by other installed packages under the `qml_observer.detectors`
  entry-point group. Plugins run **in-process with no sandboxing** -- a
  malicious or buggy plugin has full code execution in your training
  process. This is a known, accepted, and explicitly documented tradeoff
  for a research tool, not a vulnerability to report. `list_detector_plugins()`
  lets you see what's registered without importing/executing anything;
  `discover_detector_plugins()`/`load_detector_plugins()` do execute each
  plugin's module-level code. Only install/enable detector plugins from
  sources you trust, the same as any other Python package.
- `WebhookAction` (Milestone 10) POSTs to a user-supplied URL. By default
  it refuses obviously internal-looking targets (`localhost`, loopback,
  link-local, and private-range addresses) as a minimal SSRF safeguard;
  `allow_internal_targets=True` opts out for local development. This is a
  literal-IP/hostname check only -- it does not resolve DNS and does not
  re-validate HTTP redirect targets, so it does not protect against DNS
  rebinding or a redirect to an internal address. See
  `qml_observer.integrations.security` for exactly what it does and does
  not cover. If you embed `WebhookAction` behind a service that accepts a
  webhook URL from an untrusted caller, do not rely on this check alone.
- The optional dashboard (Milestone 11, `qml_observer.dashboard`) has no
  authentication. `run_dashboard()` binds to `127.0.0.1` by default and
  **refuses** (raises `ValueError`) to bind any other host unless you
  explicitly pass `allow_non_loopback=True`, in which case it still warns
  on stderr every time it starts. If you opt in, anything able to reach
  that host/port can read the run's data -- including loss curves,
  gradient statistics, and diagnosis evidence, and potentially
  proprietary circuit/ansatz metadata if you've included it (see
  `docs/development/data_handling.md`). The dashboard's `/api/*` routes
  are read-only; there is no route that can mutate monitor/training
  state. See `docs/architecture/dashboard.md`.
