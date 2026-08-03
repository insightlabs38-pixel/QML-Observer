# Contributing

The canonical, always-up-to-date contributing guide is the repository's
top-level `CONTRIBUTING.md` (development setup, Python version support
policy, license rationale, detector-proposal process). This page exists
because the blueprint's documentation tree expects
`docs/development/contributing.md`; rather than duplicate and risk drift,
it stays in sync by pointing here:

- **Development setup**: see `CONTRIBUTING.md#development-setup`, or
  `development_setup.md` on this page for a slightly expanded walkthrough.
- **Python version support**: rolling two-version window, currently 3.12
  and 3.13 -- see `CONTRIBUTING.md#python-version-support-policy`.
- **License rationale (MPL-2.0)**: see `CONTRIBUTING.md#why-mpl-20`.
- **Adding a new detector**: see `adding_detectors.md` on this page.
- **Proposing a new built-in detector**: goes through an RFC
  (`docs/development/detector_rfc_template.md`); community plugin
  detectors, discovered via `qml_observer.detectors.plugins` (Milestone
  14, Issue #103), do not require this process.

## Code of conduct

See the top-level `CODE_OF_CONDUCT.md`.

## Pull request checklist

Per the blueprint's Volume XVIII "Definition of Done", a feature is not
complete until it has: implementation, unit tests, integration tests
where applicable, documentation, example usage, error handling,
performance consideration, and a `CHANGELOG.md` entry for any
user-visible change. Research features additionally need a mathematical
description, references, a validation methodology, benchmark results, and
known limitations -- see `research/validation.md` for what that looks like
in practice.
