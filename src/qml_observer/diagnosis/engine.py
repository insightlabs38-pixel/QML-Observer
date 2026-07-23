"""DiagnosisEngine: combines detector outputs into a single diagnosis.

Milestone 4 (Volume VII), Issue #29.

Per the blueprint's second architectural rule, detector outputs
(`DetectorResult`, one per detector, Volume V) must never be exposed
directly as the final diagnosis. `DiagnosisEngine.evaluate()` is the one
seam that turns "N independent per-detector opinions" into "one
explainable `DiagnosisResult`" (`schemas.diagnosis.DiagnosisResult`).

Scope note: the combination logic below (`_combine`) is deliberately
simple, deterministic, and fully explainable, per blueprint Volume VII
("The first version should be deterministic and explainable... Do not
introduce machine learning into the diagnosis engine initially"). It is
intentionally kept inline here rather than factored into a separate
`diagnosis/scoring.py` module, since extracting and generalizing the
scoring model into its own reusable primitive
(`combine_detector_results`) plus human-readable explanation templates
is explicitly Issues #30-#31 (confidence scoring, explanations) --
not yet in scope. This engine is fully functional in the meantime; #30
will refine *how* confidence is computed, not *whether* combination
happens.
"""

from __future__ import annotations

from qml_observer.core.events import StepObservation
from qml_observer.core.state import RunState
from qml_observer.detectors.base import BaseDetector, DetectorResult
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

#: Maps a detector's stable name to the IssueType it is evidence for when
#: triggered. Detectors not in this table (e.g. future third-party
#: detectors) are still included in `evidence`/`recommendations` when
#: triggered, but cannot become the engine's headline `issue` -- an
#: unrecognized detector name is deliberately not enough to steer the
#: overall diagnosis.
_ISSUE_BY_DETECTOR: dict[str, IssueType] = {
    "barren_plateau": IssueType.POSSIBLE_BARREN_PLATEAU,
    "stagnation": IssueType.STAGNATION,
    "convergence": IssueType.CONVERGED,
}

#: Confidence, per triggered issue type, at or above which the engine
#: reports "critical" severity rather than "warning". `CONVERGED` is
#: never critical -- it is good news, not a failure mode.
_CRITICAL_CONFIDENCE = 0.8


class DiagnosisEngine:
    """Drives a list of detectors and combines their output each step."""

    def __init__(self, detectors: list[BaseDetector]):
        """Create an engine over the given detectors.

        Args:
            detectors: Detectors to run, in no particular required order.
                An empty list is allowed; `evaluate()` will always return
                `INSUFFICIENT_EVIDENCE` in that case.

        Raises:
            TypeError: If `detectors` is not a list of `BaseDetector`.
        """
        if not isinstance(detectors, list):
            raise TypeError(f"detectors must be a list, got {type(detectors)!r}")
        for i, d in enumerate(detectors):
            if not isinstance(d, BaseDetector):
                raise TypeError(f"detectors[{i}] must be a BaseDetector, got {type(d)!r}")
        self._detectors = list(detectors)

    @property
    def detectors(self) -> list[BaseDetector]:
        """The detectors this engine drives (read-only view)."""
        return list(self._detectors)

    def evaluate(self, event: StepObservation, state: RunState) -> DiagnosisResult:
        """Feed `event` to every detector and combine their verdicts.

        Args:
            event: The `StepObservation` just recorded by the monitor.
            state: The run's overall rolling state, passed through to
                each detector's `update()`.

        Returns:
            A single `DiagnosisResult` combining all detector output for
            the current step. Never raises on a well-formed `event`;
            callers relying on the fail-open policy (addendum §1) should
            still wrap `evaluate()` per that policy, since individual
            detector implementations are not guaranteed exception-free.
        """
        results: list[DetectorResult] = []
        for detector in self._detectors:
            detector.update(event, state)
            results.append(detector.diagnose())
        return self._combine(results)

    def reset(self) -> None:
        """Reset every driven detector back to its construction-time state."""
        for detector in self._detectors:
            detector.reset()

    def _combine(self, results: list[DetectorResult]) -> DiagnosisResult:
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
            # CONVERGED is intentionally given priority over any other
            # simultaneously-triggered issue (e.g. POSSIBLE_BARREN_PLATEAU):
            # a stagnation-style detector can only see "the loss isn't
            # moving", which is exactly as true for "already at a good
            # optimum" as it is for "collapsed". ConvergenceDetector is the
            # one detector that additionally confirms the loss is *good* in
            # absolute terms, which is why the blueprint calls this
            # distinction "essential" -- it is the more specific, more
            # authoritative signal whenever it fires. Ties among
            # non-CONVERGED triggers are broken by highest confidence.
            converged_hits = [
                r
                for r in triggered
                if _ISSUE_BY_DETECTOR.get(r.detector_name) == IssueType.CONVERGED
            ]
            candidates = converged_hits if converged_hits else triggered
            winner = max(candidates, key=lambda r: r.confidence)
            issue = _ISSUE_BY_DETECTOR.get(winner.detector_name, IssueType.INSUFFICIENT_EVIDENCE)
            confidence = winner.confidence
            severity = self._severity_for(issue, confidence)
            recommendations = list(
                dict.fromkeys(rec for r in triggered for rec in r.recommendations)
            )
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

        # No detector triggered: healthy, with confidence reflecting how
        # close the nearest-to-triggering detector is (a near-miss lowers
        # our confidence that things are truly healthy).
        closest_confidence = max((r.confidence for r in results), default=0.0)
        healthy_confidence = round(1.0 - closest_confidence, 4)
        return DiagnosisResult(
            issue=IssueType.HEALTHY,
            confidence=healthy_confidence,
            severity="info",
            evidence=all_evidence,
            recommendations=[],
        )

    @staticmethod
    def _severity_for(issue: IssueType, confidence: float) -> str:
        if issue == IssueType.CONVERGED:
            return "info"
        return "critical" if confidence >= _CRITICAL_CONFIDENCE else "warning"
