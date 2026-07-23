"""Confidence scoring: combine detector results into a single diagnosis.

Milestone 4 (Volume VII), Issue #30.

`DiagnosisEngine.evaluate()` (Issue #29) produces one `DetectorResult` per
configured detector; `combine_detector_results()` is the standalone,
reusable primitive that turns that list into a single `DiagnosisResult`.
It is extracted out of `DiagnosisEngine` (which held this logic inline as
`_combine` during Issues #25-#29) into its own module specifically so:

- researchers can call `combine_detector_results()` directly against
  synthetic fixtures (`tests/fixtures/synthetic_runs.py`, Issue #32) or
  recorded benchmark logs, without needing a live `DiagnosisEngine` or
  `QMLMonitor`.
- alternative scoring models have a clear, swappable seam distinct from
  the engine's per-step driving logic -- per blueprint Volume VII: "Later,
  researchers may be able to contribute alternative scoring models."

Scope: this remains deterministic and fully explainable, per blueprint
Volume VII ("The first version should be deterministic and explainable...
Do not introduce machine learning into the diagnosis engine initially").

Weighted evidence combination
------------------------------
When more than one detector independently points at the *same* issue
(e.g. a future third-party detector also flags `POSSIBLE_BARREN_PLATEAU`),
agreement should increase confidence beyond what any single detector
reported alone -- this is the "Weighted Evidence" step in the blueprint's
Volume VII diagram. Same-issue confidences are combined via a noisy-OR:
treating each detector's confidence as an independent probability that the
issue is real,

    combined = 1 - prod(1 - weight_i * confidence_i)

which reduces to exactly `confidence_i` when only one detector maps to an
issue (today's common case, with the default weight of `1.0` for every
detector) -- so this refactor changes *nothing* about Milestone 4's
existing single-detector-per-issue confidence values -- while naturally
rewarding agreement once more detectors contribute evidence for the same
issue. `weights` exists specifically so the empirical calibration process
(addendum §3) has somewhere to put a per-detector trust adjustment without
touching detector implementations themselves.
"""

from __future__ import annotations

from qml_observer.detectors.base import DetectorResult
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

#: Maps a detector's stable name to the IssueType it is evidence for when
#: triggered. Detectors not in this table (e.g. future third-party
#: detectors) are still included in `evidence`/`recommendations` when
#: triggered, but cannot become the headline `issue` -- an unrecognized
#: detector name is deliberately not enough to steer the overall
#: diagnosis.
ISSUE_BY_DETECTOR: dict[str, IssueType] = {
    "barren_plateau": IssueType.POSSIBLE_BARREN_PLATEAU,
    "stagnation": IssueType.STAGNATION,
    "convergence": IssueType.CONVERGED,
}

#: Confidence, per triggered issue type, at or above which the result is
#: reported as "critical" severity rather than "warning". `CONVERGED` is
#: never critical -- it is good news, not a failure mode.
CRITICAL_CONFIDENCE = 0.8


def combine_detector_results(
    results: list[DetectorResult],
    weights: dict[str, float] | None = None,
) -> DiagnosisResult:
    """Combine per-detector results into one explainable `DiagnosisResult`.

    Args:
        results: One `DetectorResult` per detector run this step (order
            does not affect the outcome).
        weights: Optional per-detector-name weight in `[0, inf)` used in
            the noisy-OR combination below, for detectors whose evidence
            should count for more or less than the default of `1.0` (e.g.
            during empirical calibration, addendum §3). Detector names not
            present in `weights` default to `1.0`. A weight of `0.0`
            effectively silences that detector's contribution to the
            combined confidence (its evidence/recommendations are still
            included).

    Returns:
        A single `DiagnosisResult` summarizing all detector output for the
        step these `results` were produced from.
    """
    weights = weights or {}

    if not results:
        return DiagnosisResult(
            issue=IssueType.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            severity="info",
            evidence=["No detectors configured."],
            recommendations=["Attach one or more detectors to enable diagnosis."],
        )

    all_evidence: list[str] = []
    for r in results:
        all_evidence.extend(f"[{r.detector_name}] {line}" for line in r.evidence)

    triggered = [r for r in results if r.triggered]

    if triggered:
        issue, confidence = _resolve_triggered_issue(triggered, weights)
        severity = _severity_for(issue, confidence)
        recommendations = list(dict.fromkeys(rec for r in triggered for rec in r.recommendations))
        return DiagnosisResult(
            issue=issue,
            confidence=confidence,
            severity=severity,
            evidence=all_evidence,
            recommendations=recommendations,
        )

    # Nothing triggered. If every detector has literally no data yet,
    # report INSUFFICIENT_EVIDENCE rather than a false-confidence HEALTHY.
    if all(r.confidence == 0.0 and not r.evidence for r in results):
        return DiagnosisResult(
            issue=IssueType.INSUFFICIENT_EVIDENCE,
            confidence=0.0,
            severity="info",
            evidence=all_evidence,
            recommendations=["Insufficient data collected yet to produce a diagnosis."],
        )

    # No detector triggered: healthy, with confidence reflecting how close
    # the nearest-to-triggering detector is (a near-miss lowers our
    # confidence that things are truly healthy).
    closest_confidence = max((r.confidence for r in results), default=0.0)
    healthy_confidence = round(1.0 - closest_confidence, 4)
    return DiagnosisResult(
        issue=IssueType.HEALTHY,
        confidence=healthy_confidence,
        severity="info",
        evidence=all_evidence,
        recommendations=[],
    )


def _resolve_triggered_issue(
    triggered: list[DetectorResult],
    weights: dict[str, float],
) -> tuple[IssueType, float]:
    """Pick the headline issue + combined confidence among triggered results."""
    by_issue: dict[IssueType, list[DetectorResult]] = {}
    for r in triggered:
        issue = ISSUE_BY_DETECTOR.get(r.detector_name)
        if issue is not None:
            by_issue.setdefault(issue, []).append(r)

    if not by_issue:
        # Every triggered detector is unrecognized (e.g. a third-party
        # detector with no issue mapping registered yet); an unmapped
        # trigger alone must not steer the headline diagnosis.
        return IssueType.INSUFFICIENT_EVIDENCE, 0.0

    combined_by_issue = {issue: _noisy_or(group, weights) for issue, group in by_issue.items()}

    # CONVERGED is intentionally given priority over any other candidate
    # issue: a stagnation-style detector can only see "the loss isn't
    # moving", which is exactly as true for "already at a good optimum" as
    # it is for "collapsed". ConvergenceDetector is the one detector that
    # additionally confirms the loss is *good* in absolute terms, which is
    # why the blueprint calls this distinction "essential" -- it is the
    # more specific, more authoritative signal whenever it fires. Ties
    # among any remaining candidates are broken by highest combined
    # confidence.
    if IssueType.CONVERGED in combined_by_issue:
        return IssueType.CONVERGED, combined_by_issue[IssueType.CONVERGED]

    issue = max(combined_by_issue, key=lambda k: combined_by_issue[k])
    return issue, combined_by_issue[issue]


def _noisy_or(group: list[DetectorResult], weights: dict[str, float]) -> float:
    product = 1.0
    for r in group:
        w = max(0.0, weights.get(r.detector_name, 1.0))
        product *= max(0.0, 1.0 - min(1.0, w * r.confidence))
    return round(min(1.0, max(0.0, 1.0 - product)), 4)


def _severity_for(issue: IssueType, confidence: float) -> str:
    if issue == IssueType.CONVERGED:
        return "info"
    return "critical" if confidence >= CRITICAL_CONFIDENCE else "warning"
