"""Research-grade diagnostics (Milestone 12, blueprint Volume XIII).

Everything under `qml_observer.advanced` is deliberately *not* imported
by `qml_observer.__init__`, `core.monitor`, or any detector: these are
observation-only, opt-in research utilities a user calls explicitly
(typically interactively, or when a threshold is crossed -- see plan.md
§26, "compute expensive diagnostics only when a threshold is crossed"),
not part of the always-on per-step monitoring path. Several functions
here cost `O(n_parameters)` or more loss/state evaluations per call,
which is unacceptable in the hot per-`update()` path but appropriate for
occasional, deliberate investigation of *why* a circuit is hard to train.

Currently:

- `qml_observer.advanced.geometry` (Issues #83-#87): QFIM, conditioning,
  parameter redundancy, Hessian-vector products, loss-landscape sampling
  -- local geometry diagnostics at a single parameter point.
- `qml_observer.advanced.scaling` (Issues #88-#89): `ScalingAnalyzer`,
  fitting gradient-variance-vs-qubit-count/depth trends *across* several
  runs, to check consistency with barren-plateau theory's predicted
  exponential-decay signature.

This completes Milestone 12's originally-scoped Issues #83-#89; see
`docs/research/geometry.md` for the Issue #89b Definition-of-Done writeup
covering all seven.
"""

from qml_observer.advanced.scaling import (
    ScalingAnalysisResult,
    ScalingAnalyzer,
    ScalingObservation,
    scaling_observation_from_run_summary,
)

__all__ = [
    "ScalingAnalyzer",
    "ScalingObservation",
    "ScalingAnalysisResult",
    "scaling_observation_from_run_summary",
]
