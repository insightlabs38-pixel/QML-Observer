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
- A future third-party detector plugin API (Milestone 14, not yet
  shipped) will run plugins in-process with no sandboxing. This is a
  known, accepted, and explicitly documented tradeoff for a research
  tool, not a vulnerability to report.
