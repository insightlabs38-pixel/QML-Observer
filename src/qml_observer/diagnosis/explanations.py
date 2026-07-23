"""Human-readable diagnosis explanations.

Milestone 4 (Volume VII), Issue #31.

Turns a scored `DiagnosisResult` (`diagnosis.scoring.combine_detector_results`,
Issue #30) into a plain-English, multi-line explanation suitable for CLI
output (blueprint Volume XV), run reports (Volume XII), and webhook alert
bodies (Milestone 10) -- anywhere a human needs to understand *why* the
engine reached its conclusion without reading raw `evidence`/
`recommendations` lists cold.

This module only renders text; it never changes `issue`, `confidence`, or
`severity` -- see `diagnosis.scoring` for how those are computed.
"""

from __future__ import annotations

from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType

#: One-sentence, plain-English summary per `IssueType`, used as the
#: headline of `explain()`'s output. Deliberately short -- the evidence
#: section carries the specifics.
_HEADLINES: dict[IssueType, str] = {
    IssueType.HEALTHY: "Training appears healthy; no pathology detected.",
    IssueType.CONVERGED: (
        "Training appears to have converged to a good optimum "
        "(low loss and small gradients, sustained over time)."
    ),
    IssueType.POSSIBLE_BARREN_PLATEAU: (
        "Possible barren plateau: gradients have collapsed and the loss "
        "is not improving. This is not definitive proof of a barren "
        "plateau -- see evidence below."
    ),
    IssueType.STAGNATION: (
        "Training appears stagnant: the optimizer does not appear to be "
        "making progress (loss, parameters, and/or learning rate are frozen)."
    ),
    IssueType.NOISE_DOMINATED: (
        "Gradient signal appears dominated by statistical/shot noise "
        "rather than a genuine training signal."
    ),
    IssueType.UNSTABLE: ("Training appears numerically unstable (diverging or non-finite values)."),
    IssueType.INSUFFICIENT_EVIDENCE: (
        "Not enough data has been collected yet to produce a diagnosis."
    ),
}


def explain(diagnosis: DiagnosisResult, *, max_evidence: int | None = 5) -> str:
    """Render a `DiagnosisResult` as a short, human-readable explanation.

    Args:
        diagnosis: The scored result to explain.
        max_evidence: Maximum number of evidence lines to include, to keep
            CLI/report output readable. Pass `None` to include every
            evidence line unconditionally (e.g. for a full run report
            rather than a terminal summary). The full list always remains
            available on `diagnosis.evidence` regardless of this limit.

    Returns:
        A multi-line plain-text explanation: headline; confidence and
        severity; a degraded-mode warning if applicable (addendum §1);
        evidence (bulleted, optionally truncated); and recommendations
        (bulleted) -- in that order. Sections with nothing to show are
        omitted entirely.

    Raises:
        ValueError: If `max_evidence` is negative.
    """
    if max_evidence is not None and max_evidence < 0:
        raise ValueError(f"max_evidence must be >= 0 or None, got {max_evidence}")

    lines = [
        _HEADLINES.get(diagnosis.issue, f"Status: {diagnosis.issue.value}"),
        f"Confidence: {diagnosis.confidence:.0%} (severity: {diagnosis.severity}).",
    ]

    if diagnosis.degraded:
        reason = diagnosis.degraded_reason or "an internal component failed"
        lines.append(f"\u26a0 DIAGNOSIS DEGRADED \u2014 see logs ({reason}).")

    if diagnosis.evidence:
        shown = diagnosis.evidence if max_evidence is None else diagnosis.evidence[:max_evidence]
        lines.append("")
        lines.append("Evidence:")
        lines.extend(f"  - {line}" for line in shown)
        omitted = len(diagnosis.evidence) - len(shown)
        if omitted > 0:
            lines.append(f"  ... and {omitted} more (see diagnosis.evidence for the full list).")

    if diagnosis.recommendations:
        lines.append("")
        lines.append("Recommended next steps:")
        lines.extend(f"  - {rec}" for rec in diagnosis.recommendations)

    return "\n".join(lines)
