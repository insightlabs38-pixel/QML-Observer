# Detector RFC template

This template is for proposing a new **built-in** detector to ship inside
`qml_observer.detectors` itself. If you've written a detector that lives
in your own package instead, you don't need this process at all -- see
[`development/plugin_api.md`](plugin_api.md)'s "Detector plugins" section
and just register it under the `qml_observer.detectors` entry-point
group (Milestone 14, Issue #103). This RFC process exists only for
detectors proposed to become part of the project's own maintained,
built-in set.

Open a GitHub issue (label `type:research`, `area:detectors`) using this
template as the body.

---

## 1. Failure mode

What training pathology does this detect? How is it distinct from what
the existing built-in detectors (`BarrenPlateauDetector`,
`StagnationDetector`, `ConvergenceDetector`, `NoiseDetector`) already
cover? A new detector should identify a genuinely separate failure mode,
not a variation on threshold values for an existing one (propose a config
change or a new default instead, per addendum §3's calibration process,
for that case).

## 2. Signals used

Which of the core signals (blueprint Volume III/plan.md §6: gradient
norm/variance/SNR, loss slope/improvement, parameter update magnitude,
persistence/patience, shot count, circuit depth vs. qubit count, ...)
does this detector read, and why those specifically? Per the blueprint's
"Important: a small gradient alone should never be enough to..." rule
(applied consistently across `BarrenPlateauDetector` and
`ConvergenceDetector`), a detector combining multiple signals is strongly
preferred over one relying on a single threshold.

## 3. Distinguishing from adjacent diagnoses

The blueprint's second core architectural rule is separating detection
from diagnosis, and its third is making false positives/negatives
explicit rather than assumed. Concretely:

- What existing `IssueType`s could this detector's trigger condition be
  confused with (e.g. a new detector for "insufficient shot budget" needs
  to not fire on the same signal pattern `NoiseDetector` already covers)?
- How does `diagnosis/scoring.py::combine_detector_results()` need to
  change (if at all) so this detector's evidence interacts correctly with
  the others, rather than just being appended to the evidence list
  independently?

## 4. Mathematical description

The precise formula/statistic this detector computes, with references if
it's drawn from published work (e.g. a specific barren-plateau or
QFIM-conditioning result). Per the blueprint's Volume XVIII "Definition of
Done" for research features, this is a hard requirement, not optional
documentation to add later.

## 5. Default thresholds and calibration plan

Per addendum §3: proposed defaults are placeholders until calibrated
against the benchmark suite (`benchmarks/run_benchmarks.py`'s fixture
categories: healthy, convergence, artificial plateau, noise-dominated,
depth-scaling). State:

- Proposed starting thresholds/patience, with rationale (even if
  provisional).
- Which existing benchmark fixtures this detector should be evaluated
  against, and whether a new fixture category is needed to exercise its
  specific failure mode.
- Target false-positive rate on the healthy/convergence fixtures (the
  project's existing bar is `< 5%`, per Milestone 7's calibration; match
  or explicitly justify deviating).

## 6. Numerical edge cases

Per addendum §7: how does this detector handle zero/one observations,
NaN/Inf inputs, zero-variance/degenerate gradients, and empty arrays? Each
should degrade to a clear, non-crashing result (`None`/`nan` propagation,
or an `UNSTABLE`-consistent signal for NaN/Inf) rather than raising.

## 7. Validation methodology and known limitations

How will this detector's claims be validated beyond the benchmark suite
(e.g. against a known-difficult circuit family)? What are its known
failure modes/limitations -- circuit regimes or scales where it's
expected to under- or over-trigger?

## 8. Definition of Done checklist

- [ ] Implementation (`detectors/<name>.py`, `BaseDetector` subclass)
- [ ] Unit tests (`tests/unit/detectors/test_<name>.py`)
- [ ] Integration into a `tests/fixtures/synthetic_runs.py` scenario if it
      covers a genuinely new failure mode
- [ ] Documentation (`docs/detectors/<name>.md`)
- [ ] Example usage
- [ ] Error handling (addendum §1/§7 fail-open + numerical edge cases)
- [ ] Performance consideration (no expensive computation on every step
      by default; see plan.md §26)
- [ ] `CHANGELOG.md` entry
- [ ] Mathematical description, references, validation methodology,
      benchmark results, and known limitations, per Volume XVIII's
      research-feature bar
