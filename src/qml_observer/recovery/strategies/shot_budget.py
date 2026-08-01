"""ShotBudgetAdjustmentStrategy.

Milestone 13, Issue #93 ("Shot-budget adjustment strategy").

Applies only to `IssueType.NOISE_DOMINATED` -- the diagnosis Milestone 9's
`NoiseDetector` reports precisely when a gradient estimate is
statistically indistinguishable from shot noise, as opposed to genuinely
small (`docs/detectors/barren_plateau.md` / `docs/detectors/noise.md`
already document that distinction; this strategy is the natural recovery
counterpart to it, not a re-derivation).

Concrete shot-count math
-------------------------
Shot noise on an expectation-value estimate scales as
``uncertainty ~ 1 / sqrt(shots)`` (`statistics.snr.
estimate_measurement_uncertainty`). To move the SNR from an observed
value ``snr_now`` up to a target ``snr_target``, since SNR is inversely
proportional to uncertainty and uncertainty is inversely proportional to
``sqrt(shots)``:

    snr_target / snr_now = sqrt(shots_target / shots_now)
    shots_target = shots_now * (snr_target / snr_now) ** 2

`snr_target` defaults to the `NoiseDetector` default `snr_threshold`
(`1.0`, see `detectors/noise.py`) plus a small safety margin, unless the
caller configures this strategy with a different target. See
`docs/architecture/recovery.md` for worked examples and known limitations
(e.g. this assumes per-shot variance stays roughly constant as the shot
count changes, which holds for a fixed circuit/observable but not if the
optimizer has also moved to a very different point in parameter space).
"""

from __future__ import annotations

import math

from qml_observer.recovery.base import RecoveryContext, RecoveryRecommendation, RecoveryStrategy
from qml_observer.schemas.diagnosis import DiagnosisResult, IssueType
from qml_observer.statistics.snr import estimate_gradient_snr, estimate_measurement_uncertainty

#: Matches NoiseDetector's own default snr_threshold (detectors/noise.py),
#: plus a small margin so the recommended shot count clears the threshold
#: rather than landing exactly on its boundary.
_DEFAULT_TARGET_SNR = 1.5

#: Hard cap on the proposed multiplicative shot-count increase, so a
#: pathologically low observed SNR (e.g. near-zero) can't propose an
#: absurd, budget-destroying shot count.
_MAX_SHOT_MULTIPLIER = 100.0

#: Shot count proposed when the current one is unknown -- a commonly used
#: default for NISQ-era shot-based execution, not a claim about any
#: specific backend's ideal budget.
_FALLBACK_SHOTS = 4096


class ShotBudgetAdjustmentStrategy(RecoveryStrategy):
    """Recommends increasing the shot budget to clear the shot-noise floor."""

    name = "shot_budget_adjustment"

    def __init__(self, target_snr: float = _DEFAULT_TARGET_SNR) -> None:
        """Configure the strategy.

        Args:
            target_snr: The gradient SNR (see `statistics.snr.
                estimate_gradient_snr`) this strategy aims for when
                proposing a new shot count. Defaults to a value just
                above `NoiseDetector`'s own default `snr_threshold`.

        Raises:
            ValueError: If `target_snr <= 0`.
        """
        if target_snr <= 0:
            raise ValueError(f"target_snr must be > 0, got {target_snr}")
        self._target_snr = target_snr

    def applies_to(self, diagnosis: DiagnosisResult) -> bool:
        return diagnosis.issue is IssueType.NOISE_DOMINATED

    def propose(
        self, diagnosis: DiagnosisResult, context: RecoveryContext
    ) -> RecoveryRecommendation | None:
        current_shots = context.shots
        gradient = context.gradient

        if current_shots is None or current_shots <= 0 or gradient is None:
            # No shot/gradient context to compute a concrete target from
            # (e.g. this diagnosis was reconstructed without the
            # originating step's full observation) -- fall back to a
            # generic, guidance-only recommendation rather than guessing
            # a specific multiplier.
            rationale = [
                f"Diagnosis: noise-dominated (confidence {diagnosis.confidence:.2f}).",
                "Current shot count and/or gradient snapshot unavailable; "
                f"proposing a generic starting shot budget ({_FALLBACK_SHOTS}) "
                "rather than a computed multiplier.",
            ]
            priority = min(0.5, 0.25 + 0.2 * diagnosis.confidence)
            return RecoveryRecommendation(
                strategy_name=self.name,
                description=(
                    f"Increase shot budget to {_FALLBACK_SHOTS} (no prior shot data available)."
                ),
                priority=priority,
                parameters={"shots": _FALLBACK_SHOTS},
                rationale=rationale,
                hook_name="set_shots",
            )

        uncertainty = estimate_measurement_uncertainty(gradient.variance, current_shots)
        current_snr = estimate_gradient_snr(gradient.mean_abs, uncertainty)

        if not math.isfinite(current_snr) or current_snr <= 0:
            # SNR is 0 (fully degenerate estimate) or otherwise unusable;
            # apply the multiplier cap directly rather than dividing by a
            # non-positive/non-finite ratio.
            multiplier = _MAX_SHOT_MULTIPLIER
        else:
            multiplier = min(_MAX_SHOT_MULTIPLIER, (self._target_snr / current_snr) ** 2)
        # Never propose *fewer* shots than currently used -- this strategy
        # only ever recommends increasing the budget.
        multiplier = max(multiplier, 1.0)
        new_shots = max(current_shots + 1, round(current_shots * multiplier))

        rationale = [
            f"Diagnosis: noise-dominated (confidence {diagnosis.confidence:.2f}).",
            f"Current shot count {current_shots}; estimated gradient SNR "
            f"{current_snr:.3f} (target {self._target_snr:.3f}).",
            f"Shot-noise uncertainty scales as 1/sqrt(shots), so reaching the "
            f"target SNR needs roughly {multiplier:.2f}x the current shot count.",
        ]
        priority = min(1.0, 0.55 + 0.35 * diagnosis.confidence)
        return RecoveryRecommendation(
            strategy_name=self.name,
            description=f"Increase shot budget from {current_shots} to {new_shots}.",
            priority=priority,
            parameters={"shots": new_shots},
            rationale=rationale,
            hook_name="set_shots",
        )
