"""Quantum-geometry research diagnostics (Milestone 12, Issues #83-#87).

Explains *why* a circuit is hard to train, not just that it is (blueprint
Volume XIII / `future_milestones_plan.md` Milestone 12), by estimating the
local optimization geometry around a given parameter point:

- `qfim.py` (Issue #83): quantum Fisher information matrix (QFIM)
  estimation from a user-supplied statevector function.
- `conditioning.py` (Issue #84): QFIM condition number and effective
  rank, built on `qfim.py`.
- `redundancy.py` (Issue #85): parameter-redundancy detection in the
  ansatz, built on `conditioning.py`.
- `hessian.py` (Issue #86): finite-difference Hessian-vector product
  estimation, observation-only (no analytic second-order machinery).
- `loss_landscape.py` (Issue #87): 1D/2D loss-landscape sampling
  utilities along arbitrary directions in parameter space.

Every function in this subpackage is pure and stateless (no `BaseDetector`
subclasses here): Milestone 12's own scope is *diagnostics*, not new
autonomous detectors, per the blueprint's Volume XIII listing. A future
detector could be built on top of these (e.g. a redundancy-aware
`BarrenPlateauDetector` variant), but that is out of scope for the first
five issues.
"""

from qml_observer.advanced.geometry.conditioning import (
    ConditioningResult,
    effective_rank,
    qfim_condition_number,
    summarize_conditioning,
)
from qml_observer.advanced.geometry.hessian import estimate_hessian_vector_product
from qml_observer.advanced.geometry.loss_landscape import (
    LandscapeSample,
    landscape_flatness,
    random_direction,
    sample_loss_landscape_1d,
    sample_loss_landscape_2d,
)
from qml_observer.advanced.geometry.qfim import estimate_qfim
from qml_observer.advanced.geometry.redundancy import (
    RedundancyResult,
    detect_redundant_parameters,
)

__all__ = [
    "estimate_qfim",
    "qfim_condition_number",
    "effective_rank",
    "ConditioningResult",
    "summarize_conditioning",
    "RedundancyResult",
    "detect_redundant_parameters",
    "estimate_hessian_vector_product",
    "LandscapeSample",
    "sample_loss_landscape_1d",
    "sample_loss_landscape_2d",
    "random_direction",
    "landscape_flatness",
]
